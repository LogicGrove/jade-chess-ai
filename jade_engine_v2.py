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
