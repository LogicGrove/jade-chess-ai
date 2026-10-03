"""Jade 1.0: Stockfish MultiPV + Boltzmann + frecuencias humanas persistentes.

Código que se incluye íntegro en Jade_Colab.ipynb. No necesita este archivo aparte.
"""
import gzip
import hashlib
import html
import io
import json
import math
import os
import shutil
import sqlite3
import tempfile
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import chess
import chess.engine
import chess.pgn
import chess.svg
import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import HfApi, HfFileSystem
import ipywidgets as widgets
from IPython.display import display

PROFILES = (1200, 1300, 1400, 1500, 1600)
EVENTS = {"rapid": "Rated Rapid game", "blitz": "Rated Blitz game"}
REPO = "Lichess/standard-chess-games"
MAX_PLIES = 100


def position_key(board):
    # Comparte frecuencias entre transposiciones; el motor sí recibe el historial.
    return " ".join(board.fen(en_passant="legal").split()[:4])


def rating_band(rating):
    return 100 * math.floor((int(rating) + 50) / 100)


class JadeMemory:
    """SQLite evita mantener millones de posiciones en la RAM de Python."""
    def __init__(self, path, rhythm="rapid"):
        if rhythm not in EVENTS:
            raise ValueError("El ritmo debe ser rapid o blitz.")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.rhythm = rhythm
        self.conn = sqlite3.connect(str(self.path))
        self.conn.execute("PRAGMA journal_mode=DELETE")
        self.conn.execute("PRAGMA synchronous=FULL")
        self.conn.execute("PRAGMA cache_size=-32000")
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
            CREATE TABLE IF NOT EXISTS counts (
                profile INTEGER, position TEXT, move TEXT, n INTEGER NOT NULL,
                PRIMARY KEY(profile, position, move)) WITHOUT ROWID;
            CREATE TABLE IF NOT EXISTS games (id TEXT PRIMARY KEY) WITHOUT ROWID;
            CREATE TABLE IF NOT EXISTS cursors (
                source TEXT PRIMARY KEY, row_group INTEGER, row_offset INTEGER,
                done INTEGER DEFAULT 0);
            CREATE TABLE IF NOT EXISTS profile_stats (
                profile INTEGER PRIMARY KEY, games INTEGER, moves INTEGER);
        """)
        config = {"schema": 1, "rhythm": rhythm, "profiles": list(PROFILES),
                  "max_plies": MAX_PLIES, "position_key": "fen4-legalep"}
        previous = self.get_meta("config")
        if previous is not None and previous != config:
            self.conn.close()
            raise ValueError("La memoria tiene otra configuración. Usa una carpeta nueva.")
        with self.conn:
            self.set_meta("config", config)

    def get_meta(self, key, default=None):
        row = self.conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def set_meta(self, key, value):
        self.conn.execute("INSERT OR REPLACE INTO meta VALUES (?,?)",
                          (key, json.dumps(value, ensure_ascii=False)))

    def cursor(self, source):
        row = self.conn.execute(
            "SELECT row_group,row_offset,done FROM cursors WHERE source=?",
            (source,)).fetchone()
        return tuple(row) if row else (0, 0, 0)

    def set_cursor(self, source, group, offset, done=False):
        self.conn.execute("INSERT OR REPLACE INTO cursors VALUES (?,?,?,?)",
                          (source, int(group), int(offset), int(done)))

    def ingest(self, row):
        """Se invoca dentro de la transacción que también guarda el cursor."""
        if row.get("Event") != EVENTS[self.rhythm]:
            return "otro_ritmo"
        if row.get("WhiteTitle") == "BOT" or row.get("BlackTitle") == "BOT":
            return "bot"
        try:
            ratings = {chess.WHITE: int(row["WhiteElo"]),
                       chess.BLACK: int(row["BlackElo"])}
        except (KeyError, TypeError, ValueError, OverflowError):
            return "rating_invalido"
        if not any(rating_band(e) in PROFILES for e in ratings.values()):
            return "otro_rating"
        text = row.get("movetext")
        site = row.get("Site")
        if not isinstance(text, str) or not text.strip() or not isinstance(site, str) or not site:
            return "datos_incompletos"
        # Site es el identificador público de la partida; no guardamos jugadores.
        game_id = site.rstrip("/")
        if self.conn.execute("SELECT 1 FROM games WHERE id=?", (game_id,)).fetchone():
            return "duplicada"
        try:
            game = chess.pgn.read_game(io.StringIO(text))
        except (ValueError, IndexError):
            return "pgn_invalido"
        if game is None or game.errors or game.board().fen() != chess.STARTING_FEN:
            return "pgn_invalido"
        board = game.board()
        observations, per_profile = Counter(), Counter()
        for ply, move in enumerate(game.mainline_moves()):
            if ply >= MAX_PLIES:
                break
            band = rating_band(ratings[board.turn])
            if band in PROFILES:
                observations[(band, position_key(board), move.uci())] += 1
                per_profile[band] += 1
            board.push(move)
        if not observations:
            return "sin_movimientos"
        self.conn.execute("INSERT INTO games VALUES (?)", (game_id,))
        self.conn.executemany("""
            INSERT INTO counts VALUES (?,?,?,?)
            ON CONFLICT(profile,position,move) DO UPDATE SET n=n+excluded.n
        """, ((*key, count) for key, count in observations.items()))
        self.conn.executemany("""
            INSERT INTO profile_stats VALUES (?,1,?)
            ON CONFLICT(profile) DO UPDATE SET games=games+1, moves=moves+excluded.moves
        """, per_profile.items())
        self.set_meta("accepted_games", self.get_meta("accepted_games", 0) + 1)
        return "nueva"

    def counts_for(self, board, profile):
        return dict(self.conn.execute(
            "SELECT move,n FROM counts WHERE profile=? AND position=?",
            (rating_band(profile), position_key(board))).fetchall())

    def summary(self):
        rows = self.conn.execute("SELECT profile,games,moves FROM profile_stats ORDER BY profile").fetchall()
        return {"partidas": self.get_meta("accepted_games", 0),
                "filas_examinadas": self.get_meta("scanned_rows", 0),
                "perfiles": {int(e): {"partidas": g, "movimientos": m} for e, g, m in rows},
                "base_MB": round(self.path.stat().st_size / 1024**2, 2)}

    def close(self):
        self.conn.close()


def restore_memory(local_path, backup_dir, rhythm):
    """Restaura la copia completa más reciente; ignora archivos .part incompletos."""
    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    if local_path.exists() or backup_dir is None:
        return
    backups = sorted(Path(backup_dir).glob(f"jade_{rhythm}_*.sqlite.gz"), reverse=True)
    errors = []
    for backup in backups:
        temp = local_path.with_suffix(".restore")
        try:
            with gzip.open(backup, "rb") as source, temp.open("wb") as target:
                shutil.copyfileobj(source, target)
            conn = sqlite3.connect(str(temp))
            try:
                valid = conn.execute("PRAGMA quick_check").fetchone()[0] == "ok"
                config = json.loads(conn.execute("SELECT value FROM meta WHERE key='config'").fetchone()[0])
                valid = valid and config["rhythm"] == rhythm and config["schema"] == 1
            finally:
                conn.close()
            if not valid:
                raise ValueError("Copia no válida")
            os.replace(temp, local_path)
            print("Memoria recuperada:", backup.name)
            return
        except (OSError, ValueError, EOFError, sqlite3.Error, TypeError) as exc:
            errors.append(f"{backup.name}: {exc}")
            temp.unlink(missing_ok=True)
    if backups:
        raise RuntimeError("No se pudo recuperar ninguna copia. " + "; ".join(errors))


def save_memory(memory, backup_dir):
    """Snapshot consistente. Drive recibe un solo archivo comprimido por guardado."""
    if memory.conn.in_transaction:
        raise RuntimeError("Hay una transacción abierta; no se puede guardar todavía.")
    target_dir = Path(backup_dir) if backup_dir is not None else memory.path.parent / "copias"
    target_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    name = f"jade_{memory.rhythm}_{stamp}.sqlite.gz"
    final = target_dir / name
    part = target_dir / (name + ".part")
    with tempfile.TemporaryDirectory() as folder:
        snapshot = Path(folder) / "memory.sqlite"
        destination = sqlite3.connect(str(snapshot))
        try:
            memory.conn.backup(destination)
        finally:
            destination.close()
        compressed = Path(folder) / name
        with snapshot.open("rb") as source, gzip.open(compressed, "wb", compresslevel=3) as dest:
            shutil.copyfileobj(source, dest)
        shutil.copyfile(compressed, part)
        if part.stat().st_size != compressed.stat().st_size:
            raise IOError("Copia incompleta: se conserva el guardado anterior.")
        os.replace(part, final)
    # Conserva tres generaciones del mismo ritmo. Nunca borra la memoria local.
    for old in sorted(target_dir.glob(f"jade_{memory.rhythm}_*.sqlite.gz"), reverse=True)[3:]:
        old.unlink()
    place = "Drive" if backup_dir is not None else "Colab (temporal)"
    print(f"Guardado en {place}: {final.name} ({final.stat().st_size / 1024**2:.1f} MB)")
    return final


def month_files(memory, year, month):
    """Una revisión y un orden fijos permiten reanudar sin saltos ni duplicados."""
    key = f"manifest:{int(year)}-{int(month):02d}"
    manifest = memory.get_meta(key)
    if manifest is not None:
        return manifest
    api = HfApi()
    revision = memory.get_meta("dataset_revision") or api.dataset_info(REPO).sha
    folder = f"data/year={int(year)}/month={int(month):02d}"
    files = [{"path": item.path, "bytes": item.size}
             for item in api.list_repo_tree(REPO, path_in_repo=folder,
                                           repo_type="dataset", revision=revision)
             if item.path.endswith(".parquet")]
    if not files:
        raise ValueError("No hay Parquet en ese mes de la revisión fijada.")
    # Mezcla determinista de archivos. No pretende ser una muestra representativa.
    files.sort(key=lambda item: hashlib.sha256(("jade-v1:" + item["path"]).encode()).hexdigest())
    manifest = {"revision": revision, "files": files}
    with memory.conn:
        memory.set_meta("dataset_revision", revision)
        memory.set_meta(key, manifest)
    return manifest


def feed_jade(memory, year=2024, month=1, new_games=2000, max_rows=50_000,
              max_files=3, backup_dir=None, checkpoint_seconds=120,
              file_entries=None, log=print, quality_reviewer=None):
    """Añade partidas nuevas. Cada lote guarda contadores, IDs y cursor juntos.

    file_entries permite pruebas y Parquet locales del mismo esquema. Por defecto
    se lee Hugging Face mediante peticiones de rangos, sin descargar archivos enteros.
    """
    if min(new_games, max_rows, max_files) < 1:
        raise ValueError("Los límites deben ser positivos.")
    session = Counter()
    saved_at = time.monotonic()
    fs = None
    try:
        if file_entries is None:
            manifest = month_files(memory, year, month)
            revision = manifest["revision"]
            entries = manifest["files"]
            fs = HfFileSystem()
            log(f"Dataset: {REPO} | revisión {revision[:12]} | {year}-{month:02d}")
        else:
            entries, revision = file_entries, "local"
        opened = 0
        for entry in entries:
            name = entry["path"]
            source = f"{REPO}@{revision}/{name}"
            group, offset, done = memory.cursor(source)
            if done:
                continue
            if opened >= max_files or session["nueva"] >= new_games or session["filas"] >= max_rows:
                break
            opened += 1
            log(f"Archivo {opened}: {Path(name).name}; bloque {group}, fila {offset}")
            handle = (fs.open(f"datasets/{REPO}/{name}", "rb", revision=revision,
                              block_size=2 * 1024**2) if fs else open(name, "rb"))
            with handle:
                parquet = pq.ParquetFile(handle)
                required = ["Site", "Event", "WhiteElo", "BlackElo", "movetext"]
                missing = set(required) - set(parquet.schema_arrow.names)
                if missing:
                    raise ValueError(f"Esquema incompatible: faltan {sorted(missing)}")
                columns = required + [x for x in ("WhiteTitle", "BlackTitle")
                                      if x in parquet.schema_arrow.names]
                for rg in range(group, parquet.num_row_groups):
                    start = offset if rg == group else 0
                    total = parquet.metadata.row_group(rg).num_rows
                    batch_start = 0
                    # En modo avanzado confirma una partida por transacción para
                    # no repetir 128 análisis si el usuario detiene la ejecución.
                    batch_size=1 if quality_reviewer is not None else 128
                    for batch in parquet.iter_batches(batch_size=batch_size, row_groups=[rg], columns=columns):
                        batch_end = batch_start + batch.num_rows
                        if batch_end <= start:
                            batch_start = batch_end
                            continue
                        rows = batch.to_pylist()
                        local_start = max(0, start - batch_start)
                        delta = Counter()
                        next_offset = batch_start + local_start
                        # Una interrupción revierte tanto frecuencias como cursor de este lote.
                        with memory.conn:
                            for index in range(local_start, len(rows)):
                                if (session["nueva"] + delta["nueva"] >= new_games or
                                    session["filas"] + delta["filas"] >= max_rows):
                                    break
                                result = (memory.ingest(rows[index],quality_reviewer=quality_reviewer)
                                          if quality_reviewer is not None else memory.ingest(rows[index]))
                                delta[result] += 1
                                delta["filas"] += 1
                                next_offset = batch_start + index + 1
                            completed = next_offset >= total
                            next_group = rg + 1 if completed else rg
                            memory.set_cursor(source, next_group, 0 if completed else next_offset,
                                              done=next_group >= parquet.num_row_groups)
                            memory.set_meta("scanned_rows", memory.get_meta("scanned_rows", 0) + delta["filas"])
                        session.update(delta)
                        if quality_reviewer is not None:
                            quality_reviewer.after_commit(memory,delta)
                        batch_start = batch_end
                        if session["filas"] // 2048 != (session["filas"] - delta["filas"]) // 2048:
                            log(f"Leídas {session['filas']:,} | nuevas {session['nueva']:,}")
                        if time.monotonic() - saved_at >= checkpoint_seconds:
                            save_memory(memory, backup_dir)
                            saved_at = time.monotonic()
                        if session["nueva"] >= new_games or session["filas"] >= max_rows:
                            return dict(session)
        if opened == 0:
            log("Mes agotado. Cambia el mes para añadir otra fuente.")
        return dict(session)
    except KeyboardInterrupt:
        log("Carga interrumpida. Se conserva lo confirmado hasta el último lote.")
        return dict(session)
    finally:
        # Incluso ante un error de red, se guarda el último lote confirmado.
        try:
            save_memory(memory, backup_dir)
        finally:
            if quality_reviewer is not None:
                quality_reviewer.finish(memory)


class JadeEngine:
    def __init__(self, binary, memory, nodes=100_000):
        self.memory, self.nodes = memory, int(nodes)
        self.engine = chess.engine.SimpleEngine.popen_uci(str(binary), timeout=60)
        self.engine.configure({"Threads": 1, "Hash": 128, "Skill Level": 20,
                               "UCI_LimitStrength": False})
        self.rng = np.random.default_rng()
        self.game_token = object()

    def new_game(self):
        self.game_token = object()
        self.engine.configure({"Clear Hash": None})

    def candidates(self, board):
        if not board.is_valid():
            raise ValueError("Posición no válida.")
        if board.is_game_over():
            return []
        infos = self.engine.analyse(board, chess.engine.Limit(nodes=self.nodes),
                                    multipv=min(5, board.legal_moves.count()), game=self.game_token)
        result = []
        for info in infos:
            move = info["pv"][0]
            score = info["score"].pov(board.turn)
            result.append({"move": move, "uci": move.uci(), "san": board.san(move),
                           "cp": score.score(), "mate": score.mate(),
                           "value": score.score(mate_score=100_000)})
        return sorted(result, key=lambda row: row["value"], reverse=True)

    def distribution(self, board, candidates, profile=1300, beta=1.0, temperature=None):
        if profile not in PROFILES:
            raise ValueError("Perfil no disponible.")
        t = float(np.interp(profile, [1200, 1600], [200.0, 35.0])) if temperature is None else float(temperature)
        if not np.isfinite(t) or t <= 0 or not np.isfinite(beta) or beta < 0:
            raise ValueError("Temperatura positiva y peso humano no negativo, ambos finitos.")
        if not candidates:
            raise ValueError("No hay candidatas.")
        counts = self.memory.counts_for(board, profile)
        human = np.array([counts.get(row["uci"], 0) for row in candidates], dtype=float)
        n = human.sum()
        q = (human + 0.5) / (n + 0.5 * len(human))
        v = np.array([row["value"] for row in candidates], dtype=float)
        logits = (v - v.max()) / t + beta * (n / (n + 20.0)) * np.log(q)
        weights = np.exp(logits - logits.max())
        return weights / weights.sum(), {"temperature": t, "top5_observations": int(n),
                                          "position_observations": sum(counts.values())}

    def choose(self, board, profile=1300, beta=1.0, temperature=None):
        candidates = self.candidates(board)
        p, details = self.distribution(board, candidates, profile, beta, temperature)
        index = int(self.rng.choice(len(candidates), p=p))
        return candidates[index]["move"], details

    def close(self):
        try:
            self.engine.quit()
        except chess.engine.EngineTerminatedError:
            pass


class JadeBoard:
    """Tablero de botones nativos de Jupyter: sin servidor web ni enlace público."""
    def __init__(self, jade):
        self.jade = jade
        self.board = chess.Board()
        self.human = chess.WHITE
        self.profile, self.beta = 1300, 1.0
        self.selected, self.finished, self.busy = None, False, False
        self.closed = False
        self.result = "*"
        self.message = "Elige nivel y color; pulsa Nueva partida."
        self.last_details = None
        self.level = widgets.Dropdown(options=list(PROFILES), value=1300, description="Perfil:", layout=widgets.Layout(width="180px"))
        self.color = widgets.Dropdown(options=[("Blancas", True), ("Negras", False)], description="Juegas:", layout=widgets.Layout(width="180px"))
        self.weight = widgets.FloatSlider(value=1, min=0, max=2, step=0.25, description="Peso humano:",
                                          style={"description_width": "100px"}, layout=widgets.Layout(width="300px"))
        self.promotion = widgets.Dropdown(options=[("Dama", chess.QUEEN), ("Torre", chess.ROOK),
                         ("Alfil", chess.BISHOP), ("Caballo", chess.KNIGHT)], value=chess.QUEEN,
                         description="Promoción:", layout=widgets.Layout(width="220px"))
        self.new = widgets.Button(description="Nueva partida", button_style="success")
        self.undo = widgets.Button(description="Deshacer turno")
        self.resign = widgets.Button(description="Rendirse")
        self.draw = widgets.Button(description="Reclamar tablas")
        self.retry = widgets.Button(description="Continuar Jade")
        self.status = widgets.HTML()
        self.history = widgets.HTML()
        self.pgn = widgets.Textarea(value="", description="PGN:", layout=widgets.Layout(width="100%", height="95px"))
        self.uci = widgets.Text(placeholder="e2e4, e1g1, e7e8q", layout=widgets.Layout(width="210px"))
        self.send = widgets.Button(description="Jugar UCI", layout=widgets.Layout(width="110px"))
        self.buttons = {}
        for square in chess.SQUARES:
            button = widgets.Button(layout=widgets.Layout(width="38px", height="38px", padding="0", border="0"))
            button.style.font_size = "28px"
            button.style.font_weight = "normal"
            button.on_click(lambda _, square=square: self.click(square))
            self.buttons[square] = button
        self.grid = widgets.GridBox(layout=widgets.Layout(grid_template_columns="18px repeat(8, 38px)",
                                        grid_template_rows="repeat(8, 38px) 18px", grid_gap="0px", width="322px"))
        self.new.on_click(self.start)
        self.undo.on_click(self.take_back)
        self.resign.on_click(self.give_up)
        self.draw.on_click(self.claim_draw)
        self.retry.on_click(self.ai_turn)
        self.send.on_click(self.submit)
        title = widgets.HTML("<style>.jade-square button {color:#152e25!important;line-height:1!important;"
                             "font-family:'DejaVu Sans','Segoe UI Symbol',sans-serif!important}</style>"
                             "<div style='padding:14px;background:#113e32;color:white;border-radius:14px'>"
                             "<b style='font-size:28px'>Jade</b><br>Un rival para practicar ajedrez</div>")
        wrap = widgets.Layout(display="flex", flex_flow="row wrap", gap="6px")
        self.root = widgets.VBox([title, widgets.Box([self.level, self.color], layout=wrap), self.weight,
                widgets.HTML("<small>Los ajustes se aplican al pulsar Nueva partida. Perfiles experimentales.</small>"),
                self.new, self.status, self.grid, self.promotion,
                widgets.Box([self.uci, self.send], layout=wrap),
                widgets.Box([self.undo, self.resign, self.draw, self.retry], layout=wrap),
                self.history, widgets.Accordion(children=[self.pgn], titles=("Copiar partida PGN",))],
                layout=widgets.Layout(width="100%", max_width="540px", gap="8px"))
        for button in self.buttons.values():
            button.add_class("jade-square")
        self.rank_labels = {r: widgets.HTML(f"<small>{r+1}</small>") for r in range(8)}
        self.file_labels = {f: widgets.HTML(f"<small style='padding-left:14px'>{chess.FILE_NAMES[f]}</small>") for f in range(8)}
        self.empty_label = widgets.HTML("")
        self.render()

    def start(self, _=None):
        if self.closed or self.busy:
            return
        self.board = chess.Board()
        self.human, self.profile, self.beta = self.color.value, self.level.value, self.weight.value
        self.selected, self.finished, self.result, self.last_details = None, False, "*", None
        self.jade.new_game()
        self.message = "Toca una pieza y después su destino."
        self.render()
        if not self.human:
            self.ai_turn()

    def update_outcome(self):
        outcome = self.board.outcome()
        if outcome:
            self.finished, self.result = True, outcome.result()
            names = {"CHECKMATE": "Jaque mate", "STALEMATE": "Ahogado",
                     "INSUFFICIENT_MATERIAL": "Material insuficiente", "SEVENTYFIVE_MOVES": "75 movimientos",
                     "FIVEFOLD_REPETITION": "Repetición quíntuple"}
            self.message = names.get(outcome.termination.name, "Partida terminada") + ": " + self.result

    def render(self):
        self.update_outcome()
        ranks = range(7, -1, -1) if self.human else range(8)
        files = list(range(8)) if self.human else list(range(7, -1, -1))
        targets = {m.to_square for m in self.board.legal_moves if m.from_square == self.selected}
        last = self.board.peek() if self.board.move_stack else None
        children = []
        for rank in ranks:
            children.append(self.rank_labels[rank])
            for file in files:
                square = chess.square(file, rank)
                button = self.buttons[square]
                piece = self.board.piece_at(square)
                button.description = piece.unicode_symbol() if piece else ("·" if square in targets else "")
                button.tooltip = chess.square_name(square)
                color = "#e4eedc" if (rank + file) % 2 else "#8eae99"
                if last and square in (last.from_square, last.to_square):
                    color = "#ced688"
                if square in targets:
                    color = "#72c9a5"
                if square == self.selected:
                    color = "#f4cc72"
                if self.board.is_check() and square == self.board.king(self.board.turn):
                    color = "#f39791"
                button.style.button_color = color
                button.disabled = self.busy or self.finished or self.closed or self.board.turn != self.human
                children.append(button)
        children.append(self.empty_label)
        children.extend(self.file_labels[f] for f in files)
        self.grid.children = tuple(children)
        turn = "Tu turno" if self.board.turn == self.human else "Turno de Jade"
        evidence = ""
        if self.last_details is not None:
            n = self.last_details["top5_observations"]
            evidence = f" · {n} observaciones humanas entre las candidatas de la última respuesta"
        self.status.value = (f"<b>Perfil {self.profile} · {self.jade.memory.rhythm} · {turn}</b><br>"
                             f"{html.escape(self.message)}<br><small>{html.escape(evidence)}</small>")
        replay, notation = self.board.root(), []
        for move in self.board.move_stack:
            if replay.turn == chess.WHITE:
                notation.append(f"{replay.fullmove_number}.")
            notation.append(replay.san(move))
            replay.push(move)
        self.history.value = "<small>" + html.escape(" ".join(notation)) + "</small>"
        game = chess.pgn.Game.from_board(self.board)
        game.headers.update({"Event": "Práctica con Jade", "White": "Tú" if self.human else "Jade",
                             "Black": "Jade" if self.human else "Tú", "Result": self.result,
                             "JadeProfile": str(self.profile)})
        self.pgn.value = str(game)
        self.new.disabled = self.busy or self.closed
        self.undo.disabled = self.busy or self.closed or not self.board.move_stack
        self.resign.disabled = self.busy or self.closed or self.finished
        self.draw.disabled = self.busy or self.closed or self.finished or self.board.turn != self.human
        self.send.disabled = self.busy or self.closed or self.finished or self.board.turn != self.human
        self.retry.disabled = self.busy or self.closed or self.finished or self.board.turn == self.human

    def play_human(self, move):
        if self.closed or self.busy or self.finished or self.board.turn != self.human:
            return
        if move not in self.board.legal_moves:
            self.message = "Esa jugada no es legal."
            self.render()
            return
        self.message = "Has jugado " + self.board.san(move)
        self.board.push(move)
        self.selected = None
        self.render()
        if not self.finished:
            self.ai_turn()

    def click(self, square):
        if self.closed or self.busy or self.finished or self.board.turn != self.human:
            return
        piece = self.board.piece_at(square)
        if piece and piece.color == self.human:
            self.selected = None if square == self.selected else square
            self.render()
            return
        if self.selected is not None:
            moves = [m for m in self.board.legal_moves if m.from_square == self.selected and m.to_square == square]
            chosen = next((m for m in moves if not m.promotion or m.promotion == self.promotion.value), None)
            if chosen:
                self.play_human(chosen)
            else:
                self.message = "Elige una de las casillas verdes."
                self.render()

    def submit(self, _=None):
        try:
            move = chess.Move.from_uci(self.uci.value.strip().lower())
        except ValueError:
            self.message = "Formato UCI: e2e4; promoción: e7e8q."
            self.render()
            return
        self.uci.value = ""
        self.play_human(move)

    def ai_turn(self, _=None):
        if self.closed or self.busy or self.finished or self.board.turn == self.human:
            return
        if self.board.can_claim_draw():
            self.finished, self.result, self.message = True, "1/2-1/2", "Jade reclama tablas."
            self.render()
            return
        self.busy, self.message = True, "Jade está pensando…"
        self.render()
        try:
            move, details = self.jade.choose(self.board, self.profile, self.beta)
            self.message = "Jade juega " + self.board.san(move)
            self.last_details = details
            self.board.push(move)
        except Exception as exc:
            self.message = f"No se pudo calcular: {exc}. Prueba Continuar Jade; si el motor se cerró, ejecuta Preparar Jade."
        finally:
            self.busy = False
            self.render()

    def take_back(self, _=None):
        if self.closed or self.busy or not self.board.move_stack:
            return
        if not self.human and len(self.board.move_stack) == 1:
            self.message = "Todavía no has movido."
        else:
            self.board.pop()
            if self.board.turn != self.human and self.board.move_stack:
                self.board.pop()
            self.finished, self.result, self.selected, self.last_details = False, "*", None, None
            self.message = "Turno deshecho."
        self.render()

    def give_up(self, _=None):
        if self.closed or self.busy or self.finished:
            return
        self.result = "0-1" if self.human else "1-0"
        self.finished, self.message = True, "Te has rendido. Gana Jade."
        self.render()

    def claim_draw(self, _=None):
        if self.closed or self.busy or self.finished or self.board.turn != self.human:
            return
        if self.board.can_claim_draw():
            self.finished, self.result, self.message = True, "1/2-1/2", "Tablas reclamadas."
        else:
            self.message = "No se cumplen las condiciones para reclamar tablas."
        self.render()

    def close(self):
        self.closed = True
        self.render()


def memory_report(memory):
    summary = memory.summary()
    print(f"Jade · {memory.rhythm}: {summary['partidas']:,} partidas únicas; base local {summary['base_MB']} MB")
    print("Perfil | Partidas con ese perfil | Movimientos")
    for profile in PROFILES:
        values = summary["perfiles"].get(profile, {"partidas": 0, "movimientos": 0})
        print(f"{profile:6} | {values['partidas']:23,} | {values['movimientos']:,}")
    return summary
