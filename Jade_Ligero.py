"""Jade 2.2.1 en CPU: requiere numpy, python-chess y un Stockfish local."""
import argparse, hashlib, json, math, sqlite3, time
from pathlib import Path
import numpy as np
import chess, chess.engine
PROFILES=(1200,1300,1400,1500,1600)
JADE_FEATURES=1560
def rating_band(rating):
    return 100*math.floor((int(rating)+50)/100)
def pack_position(board):
    """Clave exacta equivalente a FEN4: piezas, turno, enroque y EP legal."""
    pieces = bytearray(32)
    for square, piece in board.piece_map().items():
        value = piece.piece_type + (0 if piece.color else 6)
        pieces[square // 2] |= value << (4 * (square % 2))
    flags = int(board.turn)
    for i, (color, kingside) in enumerate(((True, True), (True, False), (False, True), (False, False)), 1):
        allowed = board.has_kingside_castling_rights(color) if kingside else board.has_queenside_castling_rights(color)
        flags |= int(allowed) << i
    ep = board.ep_square + 1 if board.has_legal_en_passant() else 0
    return bytes(pieces) + bytes((flags, ep))

def unpack_move(value):
    return chess.Move(value & 63, (value >> 6) & 63, promotion=(value >> 12) or None)

def jade_position_id(conn,key,create=False):
    """ID de 63 bits con comparación EXACTA y resolución explícita de colisiones.

    Evita un segundo índice que repetiría los 34 bytes de cada posición. El hash
    nunca se usa como prueba de igualdad: se comprueba siempre la clave completa.
    """
    for salt in range(1000):
        pid=int.from_bytes(hashlib.blake2b(key+salt.to_bytes(4,"little"),digest_size=8).digest(),"little") & ((1<<63)-1)
        row=conn.execute("SELECT key FROM positions WHERE id=?",(pid,)).fetchone()
        if row is None:
            if not create:
                return None
            conn.execute("INSERT OR IGNORE INTO positions(id,key) VALUES (?,?)",(pid,key))
            row=conn.execute("SELECT key FROM positions WHERE id=?",(pid,)).fetchone()
        if row[0]==key:
            return pid
    raise RuntimeError("No se pudo asignar un identificador de posición sin colisión.")
def jade_context(board, profile, opponent=None, remaining=None, other_remaining=None,
                 initial=None, increment=None, rhythm="rapid", history_known=True):
    return dict(profile=profile,opponent=opponent,remaining=remaining,other_remaining=other_remaining,
                initial=initial,increment=increment,rhythm=rhythm,history_known=history_known)

def encode_jade(board, context):
    """Solo información anterior al movimiento objetivo; perspectiva del que mueve."""
    features=np.zeros(JADE_FEATURES,dtype=np.float32)
    side=board.turn
    previous=board.copy(stack=1)
    if previous.move_stack:
        previous.pop()
    for offset,position in ((0,board),(768,previous)):
        for color,base in ((side,0),(not side,6)):
            for piece_type in chess.PIECE_TYPES:
                squares=position.pieces_mask(piece_type,color)
                for square in chess.scan_forward(squares):
                    oriented=square if side else square^56
                    features[offset+(base+piece_type-1)*64+oriented]=1
    c=features[1536:]
    def known(value):
        return value is not None and math.isfinite(float(value)) and float(value)>=0
    c[0]=min(2,max(0,float(context["profile"])/2000))
    if known(context.get("opponent")):
        c[1],c[2]=min(2,max(0,float(context["opponent"])/2000)),1
    initial=context.get("initial")
    denominator=float(initial) if known(initial) and initial>0 else 600.
    for value_index,flag_index,name in ((3,4,"remaining"),(5,6,"other_remaining")):
        value=context.get(name)
        if known(value):
            c[value_index],c[flag_index]=min(4,max(0,float(value)/denominator)),1
    for value_index,flag_index,name,scale in ((7,8,"initial",7200),(9,10,"increment",60)):
        value=context.get(name)
        if known(value):
            c[value_index],c[flag_index]=min(2,math.log1p(value)/math.log1p(scale)),1
    c[11:14]=[min(board.ply()/100,3),min(board.halfmove_clock/100,2),float(board.is_repetition(2))]
    c[14:18]=[board.has_kingside_castling_rights(side),board.has_queenside_castling_rights(side),
               board.has_kingside_castling_rights(not side),board.has_queenside_castling_rights(not side)]
    c[18]=(chess.square_file(board.ep_square)+1)/8 if board.has_legal_en_passant() else 0
    c[19:24]=[context.get("rhythm")=="blitz",bool(board.move_stack),context.get("history_known",True),board.is_check(),bool(board.move_stack)]
    legal=list(board.legal_moves)
    codes=np.array([[m.from_square if side else m.from_square^56,
                     m.to_square if side else m.to_square^56,m.promotion or 0] for m in legal],dtype=np.int64).reshape(-1,3)
    return features,codes,legal

class NumpyJadePolicy:
    """No importa PyTorch ni necesita GPU. Carga solo pesos y metadatos JSON."""
    def __init__(self,weights,metadata=None):
        self.weights=weights
        self.metadata=metadata or {}
        self.alpha=float(self.metadata.get("alpha",0))
        self.profiles=set(self.metadata.get("profiles",[]))
        self.active=bool(self.metadata.get("approved",False))
    @classmethod
    def load(cls,path,metadata=None):
        with np.load(str(path),allow_pickle=False) as archive:
            weights={key:archive[key] for key in archive.files}
        return cls(weights,metadata)
    def linear(self,x,name):
        w=self.weights
        if name+".q" in w:
            value=(x@w[name+".q"].T)*w[name+".scale"]
        else:
            value=x@w[name+".weight"].T
        return value+w[name+".bias"]
    def probabilities(self,board,context):
        features,moves,legal=encode_jade(board,context)
        if not legal:
            return {}
        probabilities=self.probabilities_encoded(features,moves)
        return dict(zip((m.uci() for m in legal),probabilities.tolist()))
    def probabilities_encoded(self,features,moves):
        hidden=np.maximum(0,self.linear(np.maximum(0,self.linear(features,"fc1")),"fc2"))
        w=self.weights
        combined=np.concatenate((np.broadcast_to(hidden,(len(moves),len(hidden))),w["source.weight"][moves[:,0]],
                                  w["destination.weight"][moves[:,1]],w["promotion.weight"][moves[:,2]]),axis=1)
        logits=self.linear(np.maximum(0,self.linear(combined,"move1")),"move2").ravel().astype(np.float64)
        probabilities=np.exp(logits-logits.max());probabilities/=probabilities.sum()
        return probabilities
    def probabilities_batch(self,examples):
        """Evaluación por bloques acotados; reutiliza características ya calculadas."""
        lengths=np.array([len(x["codes"]) for x in examples])
        features=np.stack([x["features"] for x in examples])
        moves=np.concatenate([x["codes"] for x in examples])
        hidden=np.maximum(0,self.linear(np.maximum(0,self.linear(features,"fc1")),"fc2"))
        w=self.weights
        combined=np.concatenate((np.repeat(hidden,lengths,axis=0),w["source.weight"][moves[:,0]],
                                 w["destination.weight"][moves[:,1]],w["promotion.weight"][moves[:,2]]),axis=1)
        logits=self.linear(np.maximum(0,self.linear(combined,"move1")),"move2").ravel().astype(np.float64)
        result=[];offset=0
        for length in lengths:
            values=logits[offset:offset+length];p=np.exp(values-values.max());p/=p.sum()
            result.append(p);offset+=length
        return result
"""Jade 1.2: candidatas humanas y de motor, presupuesto de tiempo y muestreo."""
import math
import time
import numpy as np
import chess
import chess.engine


class JadeEngine:
    VERSION = "1.2"

    def __init__(self, binary, memory, nodes=100_000):
        self.binary, self.memory, self.nodes = str(binary), memory, int(nodes)
        self.engine = chess.engine.SimpleEngine.popen_uci(self.binary, timeout=60)
        self.engine.configure({"Threads": 1, "Hash": 128, "Skill Level": 20,
                               "UCI_LimitStrength": False})
        self.rng = np.random.default_rng()
        self.game_token = object()

    def new_game(self):
        self.game_token = object()
        self.engine.configure({"Clear Hash": None})

    @staticmethod
    def candidate_limit(seconds=None, maximum=15):
        if maximum not in (12, 15):
            raise ValueError("El máximo de candidatas debe ser 12 o 15.")
        if seconds is None or seconds > 180:
            return maximum
        if seconds > 90:
            return min(maximum, 12)
        if seconds > 30:
            return 8
        if seconds > 10:
            return 5
        return 3

    @staticmethod
    def search_seconds(remaining=None):
        # Presupuesto TOTAL de búsqueda, incluidas las comprobaciones humanas.
        if remaining is None or remaining > 180:
            return .9
        if remaining > 90:
            return .6
        if remaining > 30:
            return .35
        if remaining > 10:
            return .18
        return max(.005, min(.08, remaining * .10))

    def human_counts(self, board, profile):
        legal = {m.uci() for m in board.legal_moves}
        return {uci: float(n) for uci, n in self.memory.counts_for(board, profile).items()
                if uci in legal and n > 0}

    @staticmethod
    def _row(board, info, counts):
        move = info["pv"][0]
        score = info["score"].pov(board.turn)
        return dict(move=move, uci=move.uci(), san=board.san(move),
                    cp=score.score(), mate=score.mate(), value=score.score(mate_score=100000),
                    human_count=counts.get(move.uci(), 0), depth=info.get("depth", 0))

    def candidates(self, board, profile=1300, remaining=None, maximum=15, use_humans=True):
        if not board.is_valid():
            raise ValueError("Posición no válida.")
        if board.is_game_over():
            return []
        k = min(self.candidate_limit(remaining, maximum), board.legal_moves.count())
        counts = self.human_counts(board, profile) if use_humans else {}
        # Se reservan plazas a jugadas reales, incluso fuera del ranking del motor.
        ranked = sorted(counts, key=lambda uci: (-counts[uci], uci))
        opening = board.ply() < 24
        known = [chess.Move.from_uci(uci) for uci in ranked if counts[uci] >= 2]
        human_slots = min(k - 1, (k * 2 // 3) if opening else (k // 2))
        humans = known[:human_slots]
        budget = self.search_seconds(remaining)
        begun = time.monotonic()
        if not humans:
            infos = self.engine.analyse(board, chess.engine.Limit(time=budget, nodes=self.nodes),
                                        multipv=k, game=self.game_token)
            rows = [self._row(board, info, counts) for info in infos if info.get("pv") and "score" in info]
            for row in rows:
                row["outside_engine_list"] = False
            return sorted(rows, key=lambda row: row["value"], reverse=True)

        # Primera búsqueda libre: conserva una referencia táctica y completa plazas.
        preliminary = self.engine.analyse(board, chess.engine.Limit(time=budget * .35, nodes=max(100, self.nodes // 3)),
                                          multipv=k, game=self.game_token)
        engine_moves = [info["pv"][0] for info in preliminary if info.get("pv")]
        if not engine_moves:
            raise RuntimeError("Stockfish no devolvió candidatas.")
        pool = list(dict.fromkeys([engine_moves[0]] + humans + engine_moves))[:k]
        # Un segundo análisis compara TODAS las candidatas, incluidas las humanas.
        left = max(.005, budget - (time.monotonic() - begun))
        infos = self.engine.analyse(board, chess.engine.Limit(time=left, nodes=max(100, self.nodes * 2 // 3)),
                                    multipv=len(pool), root_moves=pool, game=self.game_token)
        rows = [self._row(board, info, counts) for info in infos if info.get("pv") and "score" in info]
        for row in rows:
            row["outside_engine_list"] = row["move"] not in engine_moves
        return sorted(rows, key=lambda row: row["value"], reverse=True)

    def distribution(self, board, candidates, profile=1300, beta=1.0, temperature=None, remaining=None):
        if profile not in (1200, 1300, 1400, 1500, 1600):
            raise ValueError("Perfil no disponible.")
        t = float(np.interp(profile, [1200, 1600], [180., 45.])) if temperature is None else float(temperature)
        if not math.isfinite(t) or t <= 0 or not math.isfinite(beta) or not 0 <= beta <= 2:
            raise ValueError("Temperatura positiva y peso humano entre 0 y 2.")
        if not candidates:
            raise ValueError("No hay candidatas.")
        # Con poco tiempo se reduce el cálculo y aumenta la dispersión, aunque
        # haya menos candidatas. No se obliga a escoger siempre la primera.
        pressure = 1. if remaining is None or remaining > 90 else 1.2 if remaining > 30 else 1.5 if remaining > 10 else 1.9
        t *= pressure
        counts = self.human_counts(board, profile)
        values = np.array([row["value"] for row in candidates], dtype=float)
        human = np.array([counts.get(row["uci"], 0) for row in candidates], dtype=float)
        loss = np.maximum(0., values.max() - values)
        weights = np.exp(np.clip((values - values.max()) / t, -700, 0))
        p_engine = weights / weights.sum()
        opening = board.ply() < 24
        # Tolerancia heurística: admite gambitos y elecciones subóptimas comunes.
        free_cp = float(np.interp(profile, [1200, 1600], [220., 100.])) * (1.25 if opening else 1.)
        penalty = np.exp(-np.maximum(0., loss - free_cp) / (120. * pressure))
        best_mate = candidates[int(np.argmax(values))]["mate"]
        for i, row in enumerate(candidates):
            if row["mate"] is not None and row["mate"] < 0 and (best_mate is None or best_mate >= 0):
                penalty[i] = 0.  # Evita un mate detectado si hay una alternativa.
            elif best_mate is not None and best_mate > 0 and row["mate"] is None:
                # No convierte un mate positivo en una diferencia ficticia de cp.
                penalty[i] = math.exp(-max(0., 350. - (row["cp"] or 0)) / 200.)
        evidence = float(human.sum())
        p_human = human * penalty
        mix = ((beta / (beta + .25)) * (evidence / (evidence + 15.)) * (1. if opening else .7)) if beta else 0.
        if p_human.sum() <= 0:
            mix, p_human = 0., p_engine.copy()
        else:
            p_human /= p_human.sum()
        probabilities = (1. - mix) * p_engine + mix * p_human
        details = dict(temperature=t, human_mix=float(mix), candidate_count=len(candidates),
                       candidate_observations=int(evidence), position_observations=sum(counts.values()),
                       # Alias para abrir partidas guardadas por la interfaz anterior.
                       top5_observations=int(evidence), opening=opening)
        return probabilities, details

    def reaction_seconds(self, board, candidates, details, remaining=None, pace=1.):
        if pace < 0 or not math.isfinite(pace):
            raise ValueError("El ritmo de respuesta debe ser positivo.")
        if board.legal_moves.count() == 1:
            target = self.rng.uniform(.4, .9)
        elif remaining is not None and remaining <= 10:
            target = self.rng.uniform(.15, .55)
        elif remaining is not None and remaining <= 30:
            target = self.rng.uniform(.4, 1.3)
        elif remaining is not None and remaining <= 90:
            target = self.rng.uniform(1., 3.)
        else:
            target = self.rng.uniform(2.5, 6.)
            close = sum(candidates[0]["value"] - row["value"] < 70 for row in candidates)
            target *= 1. + min(close, 5) * .10
            if details["opening"] and details["position_observations"] >= 30:
                target *= .75
        target = min(10., target * pace)
        if remaining is not None:
            target = min(target, max(0., remaining * .18))
        return float(target)

    def choose(self, board, profile=1300, beta=1., temperature=None, remaining=None, maximum=15, pace=1.):
        candidates = self.candidates(board, profile, remaining, maximum, use_humans=beta > 0)
        p, details = self.distribution(board, candidates, profile, beta, temperature, remaining)
        index = int(self.rng.choice(len(candidates), p=p))
        row = candidates[index]
        details.update(selected_san=row["san"], selected_human_count=row["human_count"],
                       outside_engine_list=row["outside_engine_list"], probability=float(p[index]),
                       max_candidates=self.candidate_limit(remaining, maximum))
        details["reaction_seconds"] = self.reaction_seconds(board, candidates, details, remaining, pace)
        return row["move"], details

    def close(self):
        try:
            self.engine.quit()
        except chess.engine.EngineTerminatedError:
            pass

ClassicJadeEngine=JadeEngine
"""Integra el predictor aprobado sin reemplazar Stockfish ni la memoria humana."""


class JadeEngine(ClassicJadeEngine):
    VERSION="2.2.1"

    def __init__(self,binary,memory,nodes=100000,policy=None,hash_mb=64):
        super().__init__(binary,memory,nodes)
        self.policy=policy
        self.engine.configure({"Hash":max(16,min(512,int(hash_mb)))})
        self.prediction={}
        self.prediction_key=None

    def choose(self,board,profile=1300,beta=1.,temperature=None,remaining=None,maximum=15,pace=1.):
        self.prediction={}
        self.prediction_key=(board.fen(),profile)
        if self.policy and self.policy.active and profile in self.policy.profiles and beta>0:
            context=jade_context(board,profile,remaining=remaining,rhythm=self.memory.rhythm)
            self.prediction=self.policy.probabilities(board,context)
        # Una búsqueda de una sola línea da una referencia más profunda que
        # repartir todo el cálculo entre 15 candidatas. Cuenta en el reloj real.
        self.tactical_anchor=None
        self.anchor_fen=board.fen()
        if not board.is_game_over():
            budget=self.search_seconds(remaining)*.4
            info=self.engine.analyse(board,chess.engine.Limit(time=max(.005,budget),
                                     nodes=max(100,self.nodes//2)),game=self.game_token)
            if info.get("pv") and "score" in info:
                self.tactical_anchor=self._row(board,info,{})
        move,details=super().choose(board,profile,beta,temperature,remaining,maximum,pace)
        anchor=self.tactical_anchor
        # Confirma la decisión contra la referencia con solo DOS variantes.
        # Se usa el mismo control tras TODAS las mezclas de memoria y red.
        if anchor and move!=anchor["move"]:
            budget=max(.005,self.search_seconds(remaining)*.4)
            infos=self.engine.analyse(board,chess.engine.Limit(time=budget,nodes=max(100,self.nodes//2)),
                multipv=2,root_moves=[anchor["move"],move],game=self.game_token)
            checked=[self._row(board,x,{}) for x in infos if x.get("pv") and "score" in x]
            by_move={x["move"]:x for x in checked}
            if len(by_move)==2:
                rows=list(by_move.values());weights=np.array([float(x["move"]==move) for x in rows])
                safe,guard=self.guard_distribution(rows,weights,profile)
                if safe[rows.index(by_move[move])]==0:
                    move=rows[int(np.argmax(safe))]["move"]
                    details.update(selected_san=board.san(move),probability=1.,
                                   safety_override=True,selected_human_count=self.human_counts(board,profile).get(move.uci(),0),
                                   outside_engine_list=False)
                details["verification_loss_cp"]=max(x["value"] for x in rows)-by_move[move]["value"]
                details["verified"]=True
        return move,details

    @staticmethod
    def guard_distribution(candidates,probabilities,profile):
        """Límite de pérdidas estimadas. Heurístico: NO certifica un ELO.

        No usa un porcentaje de error por movimiento que fuerce fallos sucesivos.
        Imprecisiones plausibles siguen permitidas; nunca fuerza un error.
        """
        values=np.array([r["value"] for r in candidates],dtype=float)
        loss=np.maximum(0.,values.max()-values)
        cap=float(np.interp(profile,[1200,1600],[140.,65.]))
        best=candidates[int(np.argmax(values))]
        mate_wins=lambda r:r.get("mate") is not None and r["value"]>0
        mate_loses=lambda r:r.get("mate") is not None and r["value"]<0
        allowed=loss<=cap
        for i,row in enumerate(candidates):
            if mate_wins(best):
                # Evita aplazar una y otra vez el mate por el muestreo humano.
                allowed[i]=mate_wins(row) and row["mate"]<=best["mate"]
            elif mate_loses(best):
                allowed[i]=True  # Si todas pierden, no inventa una salida.
            elif mate_loses(row):
                allowed[i]=False
        p=np.asarray(probabilities,dtype=float).copy()
        p[~allowed]=0.
        # Dentro del margen, desincentiva pérdidas continuas de medio peón.
        if not mate_wins(best) and not mate_loses(best):
            p*=np.exp(-loss/max(25.,cap*.45))
        if not np.isfinite(p).all() or p.sum()<=0:
            p=np.zeros(len(candidates));p[int(np.argmax(values))]=1.
        else:
            p/=p.sum()
        error_limit=float(np.interp(profile,[1200,1600],[.08,.03]))
        if best.get("mate") is None:
            # Una jugada muy repetida no puede convertir una pérdida apreciable
            # en la opción habitual: limita su masa conjunta, sin forzar fallos.
            error_band=loss>60.
            mass=float(p[error_band].sum())
            if mass>error_limit:
                p[error_band]*=error_limit/mass
                good_mass=float(p[~error_band].sum())
                if good_mass>0:
                    p[~error_band]*=(1-error_limit)/good_mass
                else:
                    p[int(np.argmax(values))]=1-error_limit
        return p,dict(safety_cap_cp=cap,blocked_candidates=int((~allowed).sum()),
                      error_probability_limit=error_limit,
                      expected_loss_cp=None if best.get("mate") is not None else float(p@loss))

    def _include_anchor(self,rows):
        anchor=getattr(self,"tactical_anchor",None)
        if anchor:
            # No mezcla puntuaciones de búsquedas distintas para la misma jugada.
            if all(r["move"]!=anchor["move"] for r in rows):
                anchor=dict(anchor,outside_engine_list=False)
                rows=([anchor]+rows[:-1]) if rows else [anchor]
        return sorted(rows,key=lambda r:r["value"],reverse=True)

    def candidates(self,board,profile=1300,remaining=None,maximum=15,use_humans=True):
        if getattr(self,"anchor_fen",None)!=board.fen():
            self.tactical_anchor=None
        if self.prediction_key!=(board.fen(),profile):
            self.prediction={}
        if not self.prediction or not use_humans:
            return self._include_anchor(super().candidates(board,profile,remaining,maximum,use_humans))
        if not board.is_valid():
            raise ValueError("Posición inválida.")
        if board.is_game_over():
            return []
        k=min(self.candidate_limit(remaining,maximum),board.legal_moves.count())
        counts=self.human_counts(board,profile)
        slots=min(k-1,max(1,k*2//3 if board.ply()<24 else k//2))
        observed=sorted((m for m,n in counts.items() if n>=2),key=lambda m:-counts[m])
        predicted=sorted(self.prediction,key=lambda m:-self.prediction[m])
        # Alterna datos observados y propuestas de la red. Siempre conserva la
        # primera referencia de Stockfish y vuelve a evaluar el conjunto completo.
        human=[]
        for i in range(max(len(observed),len(predicted))):
            for sequence in (observed,predicted):
                if i<len(sequence) and sequence[i] not in human:
                    human.append(sequence[i])
        human=[chess.Move.from_uci(m) for m in human[:slots]]
        budget=self.search_seconds(remaining);began=time.monotonic()
        preliminary=self.engine.analyse(board,chess.engine.Limit(time=budget*.35,nodes=max(100,self.nodes//3)),
                                        multipv=k,game=self.game_token)
        references=[x["pv"][0] for x in preliminary if x.get("pv")]
        if not references:
            raise RuntimeError("Stockfish no devolvió candidatas.")
        pool=list(dict.fromkeys([references[0]]+human+references))[:k]
        infos=self.engine.analyse(board,chess.engine.Limit(time=max(.005,budget-(time.monotonic()-began)),
                                      nodes=max(100,2*self.nodes//3)),multipv=len(pool),root_moves=pool,game=self.game_token)
        rows=[self._row(board,info,counts) for info in infos if info.get("pv") and "score" in info]
        for row in rows:
            row["outside_engine_list"]=row["move"] not in references
        return self._include_anchor(rows)

    def distribution(self,board,candidates,profile=1300,beta=1.,temperature=None,remaining=None):
        base,details=super().distribution(board,candidates,profile,beta,temperature,remaining)
        details.update(neural_active=False,neural_mix=0.)
        if (not self.prediction or not self.policy or not self.policy.active or beta<=0
            or self.prediction_key!=(board.fen(),profile)):
            base,guard=self.guard_distribution(candidates,base,profile)
            details.update(guard)
            return base,details
        p=np.array([self.prediction.get(row["uci"],0.) for row in candidates],dtype=float)
        # La red conserva preferencias humanas. El motor aplica una penalización
        # suave a pérdidas grandes; no convierte su ranking en la etiqueta humana.
        values=np.array([r["value"] for r in candidates],dtype=float)
        tolerance=float(np.interp(profile,[1200,1600],[220,100]))*(1.25 if board.ply()<24 else 1.)
        penalty=np.exp(-np.minimum(700,np.maximum(0,values.max()-values-tolerance)/180))
        best=candidates[int(np.argmax(values))]
        for i,row in enumerate(candidates):
            if best["mate"] is not None and best["mate"]>0 and row["mate"] is None:
                penalty[i]=math.exp(-max(0,350-(row["cp"] or 0))/200)
            if row["mate"] is not None and row["mate"]<0 and (best["mate"] is None or best["mate"]>=0):
                penalty[i]=0
        p*=penalty
        if p.sum()>0:
            p/=p.sum();alpha=min(.5,self.policy.alpha)*min(1.,beta)
            base=(1-alpha)*base+alpha*p
            details.update(neural_active=True,neural_mix=alpha)
        base,guard=self.guard_distribution(candidates,base,profile)
        details.update(guard)
        return base,details

class ReadOnlyMemory:
    def __init__(self,path):
        self.conn=sqlite3.connect('file:'+str(Path(path).resolve())+'?mode=ro',uri=True)
        config=json.loads(self.conn.execute("SELECT value FROM meta WHERE key='config'").fetchone()[0])
        if config['schema']!=2:
            raise ValueError('Se necesita la memoria migrada v2.')
        self.rhythm=config['rhythm']
        self.conn.execute('PRAGMA cache_size=-4000')
    def counts_for(self,board,profile):
        pid=jade_position_id(self.conn,pack_position(board))
        if pid is None:
            return {}
        return {unpack_move(code).uci():n for code,n in self.conn.execute(
            'SELECT move,n FROM counts WHERE profile=? AND position_id=?',
            (rating_band(profile),pid))}

def main():
    parser=argparse.ArgumentParser(description='Jade en CPU, movimientos UCI y tablero en consola.')
    parser.add_argument('--stockfish',required=True,help='Ruta al ejecutable de tu dispositivo')
    parser.add_argument('--memory',required=True,help='Archivo jade2_rapid.sqlite o jade2_blitz.sqlite')
    parser.add_argument('--model-dir',default='.')
    parser.add_argument('--fp32',action='store_true',help='Utiliza los pesos originales en lugar de INT8')
    parser.add_argument('--profile',type=int,choices=PROFILES,default=1300)
    parser.add_argument('--black',action='store_true',help='Jugar con negras')
    args=parser.parse_args();folder=Path(args.model_dir)
    metadata=json.loads((folder/'metadata.json').read_text())
    precision='policy_int8.npz' if not args.fp32 and metadata['quantization']['accepted'] else 'policy_fp32.npz'
    policy=NumpyJadePolicy.load(folder/precision,metadata)
    memory=ReadOnlyMemory(args.memory)
    if memory.rhythm!=metadata['rhythm']:
        raise ValueError('El modelo y la memoria son de ritmos distintos.')
    engine=JadeEngine(args.stockfish,memory,policy=policy)
    board=chess.Board();human=not args.black
    try:
        while not board.is_game_over():
            print(board.unicode(borders=True));print('Turno:', 'blancas' if board.turn else 'negras')
            if board.turn==human:
                text=input('Movimiento UCI (o salir): ').strip().lower()
                if text in ('salir','quit','exit'):
                    return
                try:
                    board.push_uci(text)
                except ValueError:
                    print('Movimiento no legal.')
            else:
                began=time.monotonic();move,detail=engine.choose(board,args.profile)
                time.sleep(max(0,detail['reaction_seconds']-(time.monotonic()-began)))
                print('Jade:',board.san(move));board.push(move)
        print(board);print(board.result())
    finally:
        engine.close();memory.conn.close()
if __name__=='__main__':
    main()
