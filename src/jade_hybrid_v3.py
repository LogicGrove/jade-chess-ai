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
