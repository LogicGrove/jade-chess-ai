"""Jade 2.2.1: predictor humano compacto, enseñanza avanzada y memoria persistente.

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
                columns = required + [x for x in ("WhiteTitle", "BlackTitle", "TimeControl")
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


LegacyJadeMemory=JadeMemory
legacy_restore_memory=restore_memory
legacy_save_memory=save_memory

"""Memoria v2: posiciones de 34 bytes, corpus separado y copias Zstandard.

Se integra después del lector Parquet original. LegacyJadeMemory y las funciones
legacy_* conservan la lectura de las copias 1.x. Nunca se migra sobre el original.
"""
import struct
import zstandard as zstd
from functools import lru_cache


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


def unpack_position(key):
    if len(key) != 34:
        raise ValueError("Clave de posición incompatible.")
    board = chess.Board(None)
    for square in chess.SQUARES:
        value = (key[square // 2] >> (4 * (square % 2))) & 15
        if value:
            board.set_piece_at(square, chess.Piece((value - 1) % 6 + 1, value <= 6))
    flags = key[32]
    board.turn = bool(flags & 1)
    for i, square in enumerate((chess.H1, chess.A1, chess.H8, chess.A8), 1):
        if flags & (1 << i):
            board.castling_rights |= chess.BB_SQUARES[square]
    board.ep_square = key[33] - 1 if key[33] else None
    return board


def pack_move(move):
    return move.from_square | (move.to_square << 6) | ((move.promotion or 0) << 12)


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


def jade_split(fingerprint):
    bucket = int(fingerprint[:8], 16) % 100
    return "train" if bucket < 80 else "validation" if bucket < 90 else "test"


class JadeMemory(LegacyJadeMemory):
    SCHEMA = 2

    def __init__(self, path, rhythm="rapid"):
        if rhythm not in EVENTS:
            raise ValueError("Ritmo no disponible.")
        self.path, self.rhythm = Path(path), rhythm
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path))
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.execute("PRAGMA synchronous=FULL")
        self.conn.execute("PRAGMA cache_size=-8000")
        existing = self.conn.execute("SELECT name FROM sqlite_master WHERE name='meta'").fetchone()
        if existing:
            config = self.get_meta("config", {})
            if config.get("schema") != 2 or config.get("rhythm") != rhythm:
                self.conn.close()
                raise ValueError("Formato antiguo u otro ritmo. Ejecuta la celda 3B de migración.")
            if self.get_meta("inference_only",False):
                self.conn.close()
                raise ValueError("Esta copia es solo para jugar. Recupera la copia completa para seguir aprendiendo.")
        self.conn.executescript('''
            CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT);
            CREATE TABLE IF NOT EXISTS positions(id INTEGER PRIMARY KEY, key BLOB NOT NULL);
            CREATE TABLE IF NOT EXISTS counts(
                profile INTEGER,position_id INTEGER,move INTEGER,n INTEGER NOT NULL,
                PRIMARY KEY(profile,position_id,move),
                FOREIGN KEY(position_id) REFERENCES positions(id)) WITHOUT ROWID;
            CREATE TABLE IF NOT EXISTS games(id TEXT PRIMARY KEY) WITHOUT ROWID;
            CREATE TABLE IF NOT EXISTS cursors(source TEXT PRIMARY KEY,row_group INTEGER,row_offset INTEGER,done INTEGER DEFAULT 0);
            CREATE TABLE IF NOT EXISTS profile_stats(profile INTEGER PRIMARY KEY,games INTEGER,moves INTEGER);
            CREATE TABLE IF NOT EXISTS corpus(
                seq INTEGER PRIMARY KEY,game_id TEXT NOT NULL UNIQUE,
                fingerprint TEXT NOT NULL UNIQUE,split TEXT NOT NULL,payload BLOB NOT NULL);
            CREATE INDEX IF NOT EXISTS corpus_split_seq ON corpus(split,seq);
            CREATE TABLE IF NOT EXISTS quality_labels(
                seq INTEGER PRIMARY KEY,fingerprint TEXT NOT NULL UNIQUE,
                split TEXT NOT NULL,labels TEXT NOT NULL);
        ''')
        with self.conn:
            self.set_meta("config", dict(schema=2,rhythm=rhythm,profiles=list(PROFILES),max_plies=MAX_PLIES,
                                         position_key="pieces34-v1",split="content-sha256-80-10-10-v1"))

    def position_id(self, key):
        return jade_position_id(self.conn,key,create=True)

    def counts_for(self, board, profile):
        pid=jade_position_id(self.conn,pack_position(board))
        if pid is None:
            return {}
        rows = self.conn.execute('''SELECT move,n FROM counts WHERE profile=? AND position_id=?''', (rating_band(profile),pid))
        return {unpack_move(move).uci(): n for move, n in rows}

    def ingest(self, row, quality_reviewer=None):
        if row.get("Event") != EVENTS[self.rhythm]:
            return "otro_ritmo"
        if row.get("WhiteTitle") == "BOT" or row.get("BlackTitle") == "BOT":
            return "bot"
        try:
            white, black = int(row["WhiteElo"]), int(row["BlackElo"])
        except (KeyError, ValueError, TypeError, OverflowError):
            return "rating_invalido"
        if not any(rating_band(x) in PROFILES for x in (white, black)):
            return "otro_rating"
        text, site = row.get("movetext"), row.get("Site")
        if not isinstance(text, str) or not isinstance(site, str) or not site:
            return "datos_incompletos"
        game_id = site.rstrip("/")
        if self.conn.execute("SELECT 1 FROM games WHERE id=?", (game_id,)).fetchone():
            return "duplicada"
        game = chess.pgn.read_game(io.StringIO(text))
        if game is None or game.errors or game.board().fen() != chess.STARTING_FEN:
            return "pgn_invalido"
        nodes = list(game.mainline())[:MAX_PLIES]
        if not nodes:
            return "sin_movimientos"
        moves = [pack_move(node.move) for node in nodes]
        # Igual secuencia, aunque cambien comentarios o URL, nunca cruza splits.
        fingerprint = hashlib.sha256(struct.pack("<" + "H" * len(moves), *moves)).hexdigest()
        if self.conn.execute("SELECT 1 FROM corpus WHERE fingerprint=?", (fingerprint,)).fetchone():
            return "duplicada_contenido"
        split = jade_split(fingerprint)
        clocks = [node.clock() for node in nodes]
        control = str(row.get("TimeControl") or game.headers.get("TimeControl", "-"))
        payload = dict(moves=moves,clocks=clocks,ratings=[black,white],time_control=control,rhythm=self.rhythm)
        packed = zstd.ZstdCompressor(level=3).compress(json.dumps(payload,separators=(",", ":")).encode())
        # Revisión ANTES de incorporar frecuencias. El test siempre queda intacto.
        labels = quality_reviewer.review(payload,fingerprint,split) if quality_reviewer and split!="test" else None
        self.conn.execute("INSERT INTO games VALUES (?)", (game_id,))
        inserted=self.conn.execute("INSERT INTO corpus(game_id,fingerprint,split,payload) VALUES (?,?,?,?)",
                                   (game_id,fingerprint,split,packed))
        if labels is not None:
            self.conn.execute("INSERT INTO quality_labels VALUES(?,?,?,?)",
                              (inserted.lastrowid,fingerprint,split,json.dumps(labels)))
        # Validación/test no alimentan ni la red ni la memoria de frecuencias.
        if split == "train":
            board, counts, stats = chess.Board(), Counter(), Counter()
            for ply,node in enumerate(nodes):
                profile = rating_band(white if board.turn else black)
                if profile in PROFILES:
                    weight=1. if labels is None else labels[str(ply)]["weight"]
                    if weight>0:
                        pid = self.position_id(pack_position(board))
                        # SQLite admite pesos REAL en n sin cambiar la memoria v2.
                        counts[(profile,pid,pack_move(node.move))] += weight
                        stats[profile] += 1
                board.push(node.move)
            self.conn.executemany('''INSERT INTO counts VALUES (?,?,?,?) ON CONFLICT(profile,position_id,move)
                DO UPDATE SET n=n+excluded.n''', ((*k,n) for k,n in counts.items()))
            self.conn.executemany('''INSERT INTO profile_stats VALUES (?,1,?) ON CONFLICT(profile)
                DO UPDATE SET games=games+1,moves=moves+excluded.moves''', stats.items())
        self.set_meta("accepted_games", self.get_meta("accepted_games", 0) + 1)
        return "nueva"

    def summary(self):
        result = super().summary()
        result["corpus"] = dict(self.conn.execute("SELECT split,COUNT(*) FROM corpus GROUP BY split"))
        result["posiciones"] = self.conn.execute("SELECT COUNT(*) FROM positions").fetchone()[0]
        result["legacy_games"] = self.get_meta("legacy_games", 0)
        return result


def jade_check_database(path, rhythm, schema=2):
    conn = sqlite3.connect("file:" + str(Path(path).resolve()) + "?mode=ro", uri=True)
    try:
        if conn.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise ValueError("Base dañada.")
        config = json.loads(conn.execute("SELECT value FROM meta WHERE key='config'").fetchone()[0])
        if config.get("schema") != schema or config.get("rhythm") != rhythm:
            raise ValueError("Formato o ritmo distinto.")
    finally:
        conn.close()


def migrate_jade_memory(old_path, new_path, rhythm="rapid", log=print):
    """Migración atómica, idempotente y sin cambiar/borrar el archivo antiguo."""
    old_path, new_path = Path(old_path), Path(new_path)
    if old_path.resolve() == new_path.resolve():
        raise ValueError("El destino debe ser otro archivo.")
    if new_path.exists():
        jade_check_database(new_path, rhythm)
        log("La memoria v2 ya existe. No se sobreescribe: " + str(new_path))
        return new_path
    if not old_path.exists():
        raise FileNotFoundError(old_path)
    jade_check_database(old_path, rhythm, schema=1)
    new_path.parent.mkdir(parents=True, exist_ok=True)
    temp = new_path.with_name(new_path.name + ".migrando-" + uuid.uuid4().hex)
    source = sqlite3.connect("file:" + str(old_path.resolve()) + "?mode=ro", uri=True)
    target = JadeMemory(temp, rhythm)
    @lru_cache(maxsize=2048)
    def convert_fen(fen):
        board = chess.Board(fen + " 0 1")
        key = pack_position(board)
        if position_key(unpack_position(key)) != fen:
            raise ValueError("Una posición no puede convertirse sin pérdida.")
        return key
    try:
        total = source.execute("SELECT COUNT(*) FROM counts").fetchone()[0]
        with target.conn:
            for table, columns in (("meta","key,value"),("games","id"),("cursors","source,row_group,row_offset,done"),
                                   ("profile_stats","profile,games,moves")):
                marks = ",".join("?" for _ in columns.split(","))
                for row in source.execute(f"SELECT {columns} FROM {table}"):
                    if table == "meta" and row[0] == "config":
                        continue
                    target.conn.execute(f"INSERT OR REPLACE INTO {table} VALUES ({marks})", row)
            for i, (profile,fen,uci,n) in enumerate(source.execute("SELECT profile,position,move,n FROM counts"),1):
                pid = target.position_id(convert_fen(fen))
                code = pack_move(chess.Move.from_uci(uci))
                if unpack_move(code).uci() != uci:
                    raise ValueError("Movimiento no convertible.")
                target.conn.execute("INSERT INTO counts VALUES (?,?,?,?)", (profile,pid,code,n))
                if i % 50000 == 0:
                    log(f"Convertidas {i:,}/{total:,} frecuencias...")
            # El cursor y las frecuencias antiguas se preservan como entrenamiento.
            target.set_meta("legacy_games", source.execute("SELECT COUNT(*) FROM games").fetchone()[0])
            target.set_meta("legacy_count_rows", total)
            target.set_meta("migration_source", old_path.name)
        old_stats = source.execute("SELECT profile,COUNT(*),SUM(n) FROM counts GROUP BY profile ORDER BY profile").fetchall()
        new_stats = target.conn.execute("SELECT profile,COUNT(*),SUM(n) FROM counts GROUP BY profile ORDER BY profile").fetchall()
        if old_stats != new_stats:
            raise ValueError("La comprobación de frecuencias no coincide.")
        for profile,fen,uci,n in source.execute("SELECT profile,position,move,n FROM counts"):
            pid=jade_position_id(target.conn,convert_fen(fen))
            found = target.conn.execute('''SELECT n FROM counts WHERE profile=? AND position_id=? AND move=?''',
                                        (profile,pid,pack_move(chess.Move.from_uci(uci)))).fetchone()
            if found != (n,):
                raise ValueError("No coincide una frecuencia migrada.")
        for table in ("games", "cursors", "profile_stats"):
            if source.execute(f"SELECT COUNT(*) FROM {table}").fetchone() != target.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone():
                raise ValueError("No coincide la tabla " + table)
        target.conn.execute("VACUUM")
        target.close()
        jade_check_database(temp,rhythm)
        os.replace(temp,new_path)
        log(f"Migración verificada: {total:,} frecuencias. El original se conserva.")
        return new_path
    finally:
        source.close()
        try:
            target.close()
        except sqlite3.Error:
            pass
        temp.unlink(missing_ok=True)


def save_memory(memory, backup_dir, level=9):
    if memory.get_meta("config",{}).get("schema")!=2:
        raise ValueError("Ejecuta 3B y 4 antes de guardar una memoria v2.")
    if memory.conn.in_transaction:
        raise RuntimeError("Termina la transacción antes de guardar.")
    folder = Path(backup_dir) if backup_dir is not None else memory.path.parent / "copias_v2"
    folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    name = f"jade2_{memory.rhythm}_{stamp}.sqlite.zst"
    final, partial = folder/name, folder/(name+".part")
    with tempfile.TemporaryDirectory() as temp:
        snapshot = Path(temp)/"snapshot.sqlite"
        dst = sqlite3.connect(str(snapshot))
        try:
            memory.conn.backup(dst)
        finally:
            dst.close()
        packed = Path(temp)/name
        with snapshot.open("rb") as src, packed.open("wb") as out:
            zstd.ZstdCompressor(level=int(level),threads=0,write_checksum=True).copy_stream(src,out)
        shutil.copyfile(packed,partial)
        # Verifica bytes por bloques antes de publicar la copia.
        def digest(path):
            h=hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda:stream.read(1024*1024),b""):
                    h.update(block)
            return h.digest()
        if digest(packed)!=digest(partial):
            raise IOError("Copia incompleta; se conservan las anteriores.")
        os.replace(partial,final)
    for old in sorted(folder.glob(f"jade2_{memory.rhythm}_*.sqlite.zst"),reverse=True)[3:]:
        old.unlink()
    print(f"Copia v2 guardada: {final.name} ({final.stat().st_size/1024**2:.2f} MB)")
    return final


def restore_memory(local_path, backup_dir, rhythm):
    local_path=Path(local_path)
    if local_path.exists():
        jade_check_database(local_path,rhythm)
        return
    if backup_dir is None:
        return
    local_path.parent.mkdir(parents=True,exist_ok=True)
    errors=[]
    for file in sorted(Path(backup_dir).glob(f"jade2_{rhythm}_*.sqlite.zst"),reverse=True):
        temp=local_path.with_name(local_path.name+".restore")
        try:
            with file.open("rb") as src,temp.open("wb") as out:
                zstd.ZstdDecompressor().copy_stream(src,out)
            jade_check_database(temp,rhythm)
            os.replace(temp,local_path)
            print("Memoria v2 recuperada:",file.name)
            return
        except (OSError,ValueError,sqlite3.Error,zstd.ZstdError) as exc:
            errors.append(str(exc))
            temp.unlink(missing_ok=True)
    if errors:
        raise RuntimeError("No se recuperó ninguna copia v2: "+"; ".join(errors))

"""Predictor humano compacto. Inferencia NumPy, PyTorch solo al entrenar.

La salida puntúa todas las jugadas legales. No contiene un vocabulario enorme
de movimientos imposibles, ni carga el corpus completo en RAM o VRAM.
"""
from contextlib import nullcontext
import copy
import gc
import random
import queue
import threading

JADE_FEATURES = 12 * 64 * 2 + 24
JADE_NET_CONFIG = dict(features=JADE_FEATURES,hidden=128,latent=64,embedding=8,move_hidden=32)


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


def make_jade_network(config=None):
    import torch
    from torch import nn
    config=dict(config or JADE_NET_CONFIG)
    class JadePolicyNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.config=config
            self.fc1=nn.Linear(config["features"],config["hidden"])
            self.fc2=nn.Linear(config["hidden"],config["latent"])
            self.source=nn.Embedding(64,config["embedding"])
            self.destination=nn.Embedding(64,config["embedding"])
            self.promotion=nn.Embedding(6,4)
            self.move1=nn.Linear(config["latent"]+2*config["embedding"]+4,config["move_hidden"])
            self.move2=nn.Linear(config["move_hidden"],1)
        def forward(self,features,moves,mask):
            hidden=torch.relu(self.fc2(torch.relu(self.fc1(features))))
            # La contribución de la posición se calcula una vez por tablero.
            # Conserva exactamente las formas/nombres de los pesos de Jade 2.0.
            latent=self.config["latent"]
            position=nn.functional.linear(hidden,self.move1.weight[:,:latent],self.move1.bias)
            moves_features=torch.cat((self.source(moves[:,:,0]),self.destination(moves[:,:,1]),self.promotion(moves[:,:,2])),dim=-1)
            move_values=nn.functional.linear(moves_features,self.move1.weight[:,latent:])
            logits=self.move2(torch.relu(position[:,None,:]+move_values)).squeeze(-1)
            return logits.float().masked_fill(~mask,float("-inf"))
    return JadePolicyNet()


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


def numpy_jade_weights(network,precision="fp32"):
    state={name:tensor.detach().cpu().float().numpy().copy() for name,tensor in network.state_dict().items()}
    if precision=="int8":
        for name in ("fc1","fc2","move1","move2"):
            w=state.pop(name+".weight")
            scale=np.maximum(np.max(np.abs(w),axis=1)/127,np.finfo(np.float32).tiny)
            state[name+".q"]=np.clip(np.rint(w/scale[:,None]),-127,127).astype(np.int8)
            state[name+".scale"]=scale
    elif precision!="fp32":
        raise ValueError("Exportación disponible: fp32 o int8.")
    return state


def jade_record_examples(payload,epoch=0,key="",positions_per_game=24,include_board=True):
    record=json.loads(zstd.ZstdDecompressor().decompress(payload))
    board=chess.Board()
    try:
        initial,increment=map(float,record["time_control"].split("+"))
        if initial<=0 or increment<0:
            initial,increment=None,None
    except (ValueError,KeyError):
        initial,increment=None,None
    remaining=[initial,initial]
    moves=record["moves"]
    rng=random.Random(hashlib.sha256((key+":"+str(epoch)).encode()).hexdigest())
    eligible=[i for i in range(len(moves)) if rating_band(record["ratings"][1-i%2]) in PROFILES]
    selected=set(rng.sample(eligible,min(len(eligible),positions_per_game)))
    for ply,code in enumerate(moves):
        move=unpack_move(code)
        side=int(board.turn)
        profile=rating_band(record["ratings"][side])
        if ply in selected and profile in PROFILES:
            context=jade_context(board,record["ratings"][side],record["ratings"][1-side],remaining[side],remaining[1-side],initial,increment,record["rhythm"])
            features,codes,legal=encode_jade(board,context)
            # Las jugadas forzadas no inflan el porcentaje de acierto del examen.
            if len(legal)>1:
                example=dict(features=features,codes=codes,target=legal.index(move),
                             context=context,uci=move.uci(),profile=profile,ply=ply)
                if include_board:
                    example["board"]=board.copy(stack=True)
                yield example
        board.push(move)
        clock=record["clocks"][ply]
        remaining[side]=float(clock) if clock is not None and math.isfinite(float(clock)) and clock>=0 else None


def jade_examples(memory,split,epoch=0,max_games=2000,positions_per_game=24,start_seq=0,include_board=True):
    if split not in ("train","validation","test"):
        raise ValueError("Split no válido.")
    rows=memory.conn.execute("SELECT seq,fingerprint,payload FROM corpus WHERE split=? AND seq>? ORDER BY seq LIMIT ?",
                             (split,start_seq,max_games))
    for seq,key,payload in rows:
        for example in jade_record_examples(payload,epoch,key,positions_per_game,include_board):
            example["seq"]=seq
            yield example


def jade_training_stream(memory,epoch,max_games,positions_per_game,start_seq,prefetch=True):
    """Prepara el siguiente lote mientras PyTorch calcula, con RAM acotada.

    El hilo tiene su propia conexión de solo lectura. No comparte la conexión
    principal ni accede a CUDA; funciona también en una celda de Colab.
    """
    if not prefetch:
        yield from jade_examples(memory,"train",epoch,max_games,positions_per_game,start_seq,include_board=False)
        return
    buffer=queue.Queue(maxsize=4);stop=threading.Event()
    def put(item):
        while not stop.is_set():
            try:
                buffer.put(item,timeout=.1);return
            except queue.Full:
                pass
    def prepare():
        conn=None
        try:
            conn=sqlite3.connect(memory.path.resolve().as_uri()+"?mode=ro",uri=True)
            rows=conn.execute("SELECT seq,fingerprint,payload FROM corpus WHERE split='train' AND seq>? ORDER BY seq LIMIT ?",
                              (start_seq,max_games))
            chunk=[]
            for seq,key,payload in rows:
                if stop.is_set():
                    return
                for example in jade_record_examples(payload,epoch,key,positions_per_game,include_board=False):
                    example["seq"]=seq;chunk.append(example)
                    if len(chunk)==64:
                        put(("data",chunk));chunk=[]
                        if stop.is_set():
                            return
            if chunk:
                put(("data",chunk))
        except BaseException as exc:
            put(("error",exc))
        finally:
            if conn is not None:
                conn.close()
            put(("end",None))
    worker=threading.Thread(target=prepare,name="jade-preparacion",daemon=True);worker.start()
    try:
        while True:
            kind,value=buffer.get(timeout=60)
            if kind=="end":
                break
            if kind=="error":
                raise value
            yield from value
    finally:
        stop.set();worker.join(timeout=2)


def jade_legacy_examples(memory,max_rows=20000):
    """Preentrenamiento opcional: frecuencias antiguas, sin inventar relojes/historia."""
    # Reservoir: muestra acotada en RAM, sin quedarse solo con el primer perfil.
    rng=random.Random(43);sample=[]
    for index,row in enumerate(memory.conn.execute('''SELECT c.profile,p.key,c.move,c.n FROM counts c
        JOIN positions p ON p.id=c.position_id''')):
        if index<max_rows:
            sample.append(row)
        else:
            selected=rng.randrange(index+1)
            if selected<max_rows:
                sample[selected]=row
    rng.shuffle(sample)
    for profile,key,code,n in sample:
        board=unpack_position(key)
        context=jade_context(board,profile,rhythm=memory.rhythm,history_known=False)
        features,codes,legal=encode_jade(board,context)
        move=unpack_move(code)
        if move in legal and len(legal)>1:
            yield dict(features=features,codes=codes,target=legal.index(move),board=board,context=context,
                       profile=profile,uci=move.uci(),ply=board.ply(),weight=float(n),seq=0)


def jade_batch(examples,device):
    import torch
    size=max(len(x["codes"]) for x in examples)
    moves=np.zeros((len(examples),size,3),dtype=np.int64)
    mask=np.zeros((len(examples),size),dtype=bool)
    for i,x in enumerate(examples):
        moves[i,:len(x["codes"])]=x["codes"];mask[i,:len(x["codes"])]=True
    return (torch.from_numpy(np.stack([x["features"] for x in examples])).to(device),
            torch.from_numpy(moves).to(device),torch.from_numpy(mask).to(device),
            torch.tensor([x["target"] for x in examples],device=device),
            torch.tensor([x.get("weight",1.) for x in examples],device=device))


def evaluate_jade_policy(policy,memory,split="validation",max_games=300,positions_per_game=24,
                         collect=False,observer=None,log=None,batch_size=64,quality=None):
    if min(max_games,positions_per_game,batch_size)<1:
        raise ValueError("El tamaño del examen y de sus lotes debe ser positivo.")
    totals=Counter();groups={};details=[];seen=set();started=time.monotonic();last_log=started
    selection=hashlib.sha256();reference=hashlib.sha256()
    def consume(examples):
        nonlocal last_log
        batch_probabilities=policy.probabilities_batch(examples)
        for example,p in zip(examples,batch_probabilities):
            target=example["target"];seen.add(example["seq"])
            side=example["board"].turn
            moves=[chess.Move(int(a) if side else int(a)^56,int(b) if side else int(b)^56,
                              promotion=int(c) or None).uci() for a,b,c in example["codes"]]
            counts=memory.counts_for(example["board"],example["profile"])
            prior=np.array([counts.get(move,0)+.5 for move in moves],dtype=float);prior/=prior.sum()
            order=np.argsort(-p,kind="stable");prior_order=np.argsort(-prior,kind="stable")
            values=dict(n=1,nll=-math.log(max(p[target],1e-12)),top1=int(order[0]==target),top3=int(target in order[:3]),
                        baseline_nll=-math.log(max(prior[target],1e-12)),baseline_top1=int(prior_order[0]==target),
                        baseline_top3=int(target in prior_order[:3]))
            if quality is not None:
                weight=float(example["weight"])
                values={k:(v if k=="n" else v*weight) for k,v in values.items()}
                values["weight_sum"]=weight
            totals.update(values)
            phase="apertura" if example["ply"]<24 else "medio/final"
            for key in (str(example["profile"]),phase,"conocida" if counts else "nueva"):
                groups.setdefault(key,Counter()).update(values)
            selection.update(example["features"].astype("<f4",copy=False).tobytes())
            selection.update(example["codes"].astype("<i8",copy=False).tobytes())
            selection.update(int(target).to_bytes(4,"little"))
            if quality is not None:
                selection.update(np.float64(weight).tobytes())
            reference.update(prior.astype("<f8",copy=False).tobytes())
            if collect:
                details.append((example,p,prior,target))
            if observer is not None:
                observer(example,p,prior,target)
        if log and time.monotonic()-last_log>=10:
            log(f"Evaluando: {totals['n']:,} posiciones · {len(seen):,} partidas con ejemplos · {time.monotonic()-started:.0f} s")
            last_log=time.monotonic()
    pending=[]
    stream=(quality.examples(memory,split,max_games=max_games) if quality is not None else
            jade_examples(memory,split,max_games=max_games,positions_per_game=positions_per_game))
    for example in stream:
        pending.append(example)
        if len(pending)>=batch_size:
            consume(pending);pending=[]
    if pending:
        consume(pending)
    def finish(values):
        n=values.get("weight_sum",values["n"])
        return {key:(value/n if key not in ("n","weight_sum") and n else value) for key,value in values.items()}
    result=finish(totals);result["groups"]={k:finish(v) for k,v in groups.items()};result["split"]=split
    result.update(n=int(totals["n"]),games_with_examples=len(seen),selection_signature=selection.hexdigest(),
                  reference_signature=reference.hexdigest(),seconds=round(time.monotonic()-started,2),
                  max_games=max_games,positions_per_game=positions_per_game)
    result["objective"]="human_quality_weighted" if quality is not None else "human_imitation"
    return (result,details) if collect else result


def jade_atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+".part")
    with temp.open("w",encoding="utf-8") as out:
        json.dump(value,out,ensure_ascii=False,indent=2);out.flush();os.fsync(out.fileno())
    os.replace(temp,path)


def load_jade_policy(folder,rhythm="rapid"):
    folder=Path(folder)
    pointer=folder/"active.json"
    if not pointer.exists():
        return None
    info=json.loads(pointer.read_text())
    directory=folder/info["directory"]
    metadata=json.loads((directory/"metadata.json").read_text())
    if metadata.get("rhythm")!=rhythm or metadata.get("features")!=JADE_FEATURES:
        raise ValueError("Modelo de otro ritmo o formato.")
    return NumpyJadePolicy.load(directory/info["weights"],metadata)


def export_jade_policy(network,memory,folder,steps,max_games=300,min_validation=200,log=print,quality=None):
    """Compara FP32 e INT8. Publica INT8 solo si pasa una prueba de fidelidad."""
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    directory=folder/("modelo_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")+"_"+uuid.uuid4().hex[:6])
    directory.mkdir()
    fp=NumpyJadePolicy(numpy_jade_weights(network));q=NumpyJadePolicy(numpy_jade_weights(network,"int8"))
    quant_nll,agreement,max_difference=0.,0,0.
    alphas=(0.,.1,.25,.4,.5)
    losses={a:0. for a in alphas}
    group_losses={};previous=load_jade_policy(folder,memory.rhythm);previous_loss=0.
    if quality is not None and previous is None:
        previous=load_jade_policy(Path(folder).parent,memory.rhythm)
    evaluated=0;last_progress=time.monotonic()
    def observe(example,p,prior,target):
        nonlocal quant_nll,agreement,max_difference,previous_loss,evaluated,last_progress
        qp=q.probabilities_encoded(example["features"],example["codes"])
        weight=float(example.get("weight",1.)) if quality is not None else 1.
        quant_nll-=weight*math.log(max(qp[target],1e-12));agreement+=int(np.argmax(qp)==np.argmax(p))
        max_difference=max(max_difference,float(np.max(np.abs(qp-p))))
        group=group_losses.setdefault(example["profile"],{a:0. for a in alphas})
        for a in alphas:
            value=-weight*math.log(max(((1-a)*prior+a*p)[target],1e-12))
            losses[a]+=value;group[a]+=value
        if previous and previous.active:
            old=previous.probabilities_encoded(example["features"],example["codes"])
            old_alpha=previous.alpha if example["profile"] in previous.profiles else 0.
            previous_loss-=weight*math.log(max(((1-old_alpha)*prior+old_alpha*old)[target],1e-12))
        evaluated+=1
        if time.monotonic()-last_progress>=10:
            log(f"Validación: {evaluated:,} posiciones revisadas; comparando candidato, INT8 y modelo anterior.")
            last_progress=time.monotonic()
    metrics=evaluate_jade_policy(fp,memory,max_games=max_games,observer=observe,quality=quality)
    n=int(metrics.get("n",0));alpha=min(losses,key=losses.get) if n else 0.
    denom=metrics.get("weight_sum",n)
    fidelity=(n>=min_validation and agreement/n>=.99 and quant_nll/max(denom,1e-12)<=metrics["nll"]+.005)
    profiles=[int(k) for k,v in metrics.get("groups",{}).items() if k.isdigit() and v["n"]>=30
              and group_losses[int(k)][alpha]<group_losses[int(k)][0.]]
    approved=bool(steps>0 and n>=min_validation and alpha>0 and profiles)
    metadata=dict(version="2.2",features=JADE_FEATURES,config=network.config,rhythm=memory.rhythm,steps=steps,
                  approved=approved,alpha=alpha,profiles=profiles,validation=metrics,
                  quantization=dict(accepted=fidelity,agreement=agreement/n if n else None,nll=quant_nll/denom if denom else None,
                                    maximum_probability_difference=max_difference),
                  objective=metrics["objective"],
                  note="Validación de predicción humana, ponderada por calidad si se indica; no certificación de ELO o fuerza de juego.")
    for name,policy in (("policy_fp32.npz",fp),("policy_int8.npz",q)):
        np.savez_compressed(directory/name,**policy.weights)
    jade_atomic_json(directory/"metadata.json",metadata)
    selected="policy_int8.npz" if fidelity else "policy_fp32.npz"
    if approved:
        # No sustituye un modelo activo por otro peor sobre la misma validación.
        candidate_loss=sum(values[alpha] if p in profiles else values[0.] for p,values in group_losses.items())
        if previous is None or not previous.active or candidate_loss<=previous_loss:
            jade_atomic_json(folder/"active.json",dict(directory=directory.name,weights=selected))
            log("Predictor aprobado y activado: "+selected)
        else:
            metadata["approved"]=False;metadata["note"]+=" Se conserva el modelo activo anterior."
            jade_atomic_json(directory/"metadata.json",metadata)
            log("El candidato no mejora al activo; se conserva el anterior.")
    else:
        log("Modelo guardado como candidato. Aún no mejora la referencia o faltan ejemplos de validación; Jade conserva su comportamiento anterior.")
    log(f"Parámetros: {sum(p.numel() for p in network.parameters()):,}; FP32 { (directory/'policy_fp32.npz').stat().st_size/1024**2:.2f} MB; INT8 {(directory/'policy_int8.npz').stat().st_size/1024**2:.2f} MB")
    return dict(directory=directory,metadata=metadata,weights=selected)


def train_jade_policy(memory,folder,epochs=2,max_games=2000,batch_size=0,effective_batch=64,
                      positions_per_game=24,device="auto",legacy_warmup=False,max_legacy=20000,
                      checkpoint_seconds=90,validation_games=300,log=print,prefetch=True,progress_seconds=10,
                      quality=None,initial_checkpoint=None):
    """Predice, calcula -log P(real) y actualiza. Reanuda pesos y optimizador.

    El checkpoint es una frontera de optimización. Una interrupción puede repetir
    el último lote/partida sin confirmar, nunca modifica validación o test.
    """
    import torch
    from torch.nn import functional as F
    if min(epochs,max_games,effective_batch,positions_per_game)<1 or batch_size<0:
        raise ValueError("Los límites de entrenamiento deben ser positivos.")
    if quality is not None:
        sync_jade_quality(memory,quality)
        if not quality.page(memory,"train",0,1) or not quality.page(memory,"validation",0,1):
            raise ValueError("Faltan partidas revisadas de train/validation. Usa 5 en modo avanzado o completa la muestra con 9A.")
        legacy_warmup=False  # No reintroduce frecuencias sin revisar en el ajuste.
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    device="cuda" if device=="auto" and torch.cuda.is_available() else "cpu" if device=="auto" else device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("Esta sesión no tiene GPU. Elige auto o cpu.")
    # Esta red pequeña suele perder tiempo al repartir matrices entre muchos hilos.
    torch.set_num_threads(1)
    batch_size=min(effective_batch,128 if device.startswith("cuda") else 64) if batch_size==0 else min(batch_size,effective_batch)
    torch.manual_seed(42)
    network=make_jade_network().to(device)
    optimizer=torch.optim.AdamW(network.parameters(),lr=3e-4,weight_decay=1e-4)
    use_amp=device.startswith("cuda")
    scaler=torch.amp.GradScaler("cuda",enabled=use_amp)
    checkpoint=folder/"training.pt"
    state=dict(epoch=0,steps=0,legacy_done=False,seq=0,rhythm=memory.rhythm)
    source=checkpoint if checkpoint.exists() else Path(initial_checkpoint) if initial_checkpoint else None
    if source is not None and source.exists():
        saved=torch.load(source,map_location=device,weights_only=True)
        if saved["state"]["rhythm"]!=memory.rhythm:
            raise ValueError("Checkpoint de otro ritmo.")
        network.load_state_dict(saved["model"]);optimizer.load_state_dict(saved["optimizer"])
        # Un checkpoint entrenado en CPU no tiene estado de GradScaler. Al pasar
        # a GPU se inicia uno nuevo; al volver a CPU no se necesita escalador.
        if use_amp and saved.get("scaler"):
            scaler.load_state_dict(saved["scaler"])
        state=saved["state"]
        if source!=checkpoint:
            state=dict(state,epoch=0,seq=0,legacy_done=True)
            log("Ajuste iniciado desde tus pesos anteriores; el checkpoint original se conserva.")
        del saved
    if quality is not None:
        for group in optimizer.param_groups:
            group["lr"]=1e-4  # Ajuste conservador sobre los pesos ya aprendidos.
    last_save=time.monotonic();interrupted=False
    def save():
        nonlocal last_save
        temporary=checkpoint.with_suffix(".pt.part")
        # No guarda modelos arbitrarios ni código ejecutable en el checkpoint.
        torch.save(dict(model=network.state_dict(),optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),state=state),temporary)
        os.replace(temporary,checkpoint);last_save=time.monotonic()
    def update(examples):
        nonlocal batch_size
        # Acumulación de gradientes: reduce VRAM manteniendo el lote efectivo.
        while True:
            optimizer.zero_grad(set_to_none=True)
            weight_sum=sum(x.get("weight",1.) for x in examples)
            total=None
            features=codes=mask=targets=weights=logits=loss=None
            try:
                for offset in range(0,len(examples),batch_size):
                    batch=examples[offset:offset+batch_size]
                    features,codes,mask,targets,weights=jade_batch(batch,device)
                    with torch.autocast(device_type="cuda",dtype=torch.float16,enabled=use_amp):
                        logits=network(features,codes,mask)
                        loss=(F.cross_entropy(logits,targets,reduction="none")*weights).sum()/weight_sum
                    scaler.scale(loss).backward()
                    total=loss.detach() if total is None else total+loss.detach()
                scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(network.parameters(),1.)
                scaler.step(optimizer);scaler.update();state["steps"]+=1
                return float(total)
            except torch.cuda.OutOfMemoryError:
                optimizer.zero_grad(set_to_none=True)
                features=codes=mask=targets=weights=logits=loss=None
                if batch_size<=1:
                    raise RuntimeError("No cabe un ejemplo en GPU. Reanuda con DISPOSITIVO_ENTRENAMIENTO='cpu'.")
                batch_size=max(1,batch_size//2);gc.collect();torch.cuda.empty_cache()
                log(f"VRAM ajustada: micro-lote {batch_size}; lote efectivo conservado.")
    try:
        if legacy_warmup and not state["legacy_done"]:
            log("Preentrenamiento con frecuencias antiguas; no se usa como examen.")
            pending=[]
            for example in jade_legacy_examples(memory,max_legacy):
                pending.append(example)
                if len(pending)>=effective_batch:
                    update(pending);pending=[]
            if pending:
                update(pending)
            state["legacy_done"]=True;save()
        for round_index in range(epochs):
            epoch=state["epoch"];pending=[];losses=[];last_seq=state["seq"]
            if quality is not None:
                page=quality.page(memory,"train",state["seq"],max_games)
                page_count=len(page);page_last=max((x[0] for x in page),default=None)
                del page
            else:
                page_count,page_last=memory.conn.execute("SELECT COUNT(*),MAX(seq) FROM (SELECT seq FROM corpus WHERE split='train' AND seq>? ORDER BY seq LIMIT ?)",
                                                         (state["seq"],max_games)).fetchone()
            began=time.monotonic();last_progress=began;examples_done=0;games_seen=0;previous_seq=None
            log(f"Ronda {round_index+1}/{epochs} · época {epoch+1} · {page_count:,} partidas · {device} · micro-lote {batch_size} / efectivo {effective_batch}.")
            stream=(quality.examples(memory,"train",epoch,max_games,state["seq"]) if quality is not None else
                    jade_training_stream(memory,epoch,max_games,positions_per_game,state["seq"],prefetch))
            try:
                for example in stream:
                    pending.append(example);last_seq=example["seq"]
                    if last_seq!=previous_seq:
                        games_seen+=1;previous_seq=last_seq
                    if len(pending)>=effective_batch:
                        losses.append(update(pending));examples_done+=len(pending);pending=[]
                        # Reinicia desde esta partida si el checkpoint cae en su mitad.
                        state["seq"]=max(0,last_seq-1)
                        now=time.monotonic()
                        if now-last_progress>=progress_seconds:
                            elapsed=now-began;eta=elapsed*max(0,page_count-games_seen)/max(1,games_seen)
                            log(f"Ronda {round_index+1}: {games_seen:,}/{page_count:,} partidas con ejemplos · {examples_done:,} posiciones · {examples_done/max(elapsed,.001):.0f} pos/s · faltan ~{eta/60:.1f} min · error {np.mean(losses[-50:]):.3f}")
                            last_progress=now
                        if now-last_save>=checkpoint_seconds:
                            save();log("Avance del entrenamiento guardado.")
            finally:
                stream.close()
            if pending:
                losses.append(update(pending))
            # Avanza incluso por partidas sin ejemplos elegibles/no forzados.
            if page_last is not None:
                last_seq=page_last
            more=(quality.page(memory,"train",last_seq,1) if quality is not None else
                  memory.conn.execute("SELECT 1 FROM corpus WHERE split='train' AND seq>? LIMIT 1",(last_seq,)).fetchone())
            if more:
                state["seq"]=last_seq
            else:
                state["seq"]=0;state["epoch"]+=1
            save()
            log(f"Ronda terminada en {time.monotonic()-began:.1f} s: {len(losses)} actualizaciones; error {np.mean(losses):.3f}" if losses else "Sin nuevas muestras de entrenamiento en el corpus.")
    except KeyboardInterrupt:
        interrupted=True;log("Entrenamiento detenido. Se conserva el último checkpoint completo.")
        # Los gradientes a medias no se guardan. Los pesos pueden ser los de la
        # última actualización completa; no se avanza el cursor de esa partida.
        optimizer.zero_grad(set_to_none=True);save()
    finally:
        network=network.cpu();del optimizer,scaler
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    log("Validando el candidato y comprobando la compresión INT8...")
    result=export_jade_policy(network,memory,folder,state["steps"],max_games=validation_games,log=log,quality=quality)
    result["interrupted"]=interrupted
    return result

"""Revisión táctica acotada y ajuste supervisado; no es aprendizaje por refuerzo.

Conserva corpus, frecuencias y red originales. Las etiquetas de calidad NO se
introducen en las características de la red. El test nunca alimenta el ajuste.
"""


def jade_quality_weight(best,played):
    """Ponderación heurística por pérdida desde el punto de vista del que mueve."""
    b=best.score(mate_score=100000);p=played.score(mate_score=100000)
    if played.is_mate() and p<0 and not (best.is_mate() and b<0):
        return 0.,None,"permite_mate"
    if best.is_mate() and b>0:
        return (1.,None,"mantiene_mate") if played.is_mate() and p>0 else (0.,None,"omite_mate")
    if best.is_mate() and b<0:
        return .25,None,"mate_inevitable"
    loss=max(0,b-p)
    weight=1. if loss<=40 else .6 if loss<=80 else .25 if loss<=140 else .05 if loss<=250 else 0.
    return weight,loss,"cp"


class JadeQuality:
    def __init__(self,path,rhythm,backup_dir=None):
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
        self.rhythm=rhythm;self.backup_dir=Path(backup_dir) if backup_dir else None
        self.conn=sqlite3.connect(str(self.path))
        self.conn.executescript('''CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY,value TEXT);
            CREATE TABLE IF NOT EXISTS reviewed(fingerprint TEXT PRIMARY KEY,seq INTEGER,split TEXT,labels TEXT);
            CREATE INDEX IF NOT EXISTS reviewed_split_seq ON reviewed(split,seq);''')
        old=self.conn.execute("SELECT value FROM settings WHERE key='rhythm'").fetchone()
        if old and old[0]!=rhythm:
            self.conn.close();raise ValueError("Revisión de calidad de otro ritmo.")
        with self.conn:
            self.conn.execute("INSERT OR IGNORE INTO settings VALUES('rhythm',?)",(rhythm,))

    def summary(self):
        return dict(self.conn.execute("SELECT split,COUNT(*) FROM reviewed GROUP BY split"))

    def save(self):
        if not self.backup_dir:
            return
        self.backup_dir.mkdir(parents=True,exist_ok=True)
        target=self.backup_dir/f"calidad_{self.rhythm}.sqlite.zst"
        with tempfile.TemporaryDirectory() as tmp:
            snapshot=Path(tmp)/"quality.sqlite";dst=sqlite3.connect(str(snapshot))
            try:
                self.conn.backup(dst)
            finally:
                dst.close()
            partial=target.with_suffix(".zst.part")
            with snapshot.open("rb") as src,partial.open("wb") as out:
                zstd.ZstdCompressor(level=6,write_checksum=True).copy_stream(src,out)
            # Confirma la copia antes de sustituir la última generación.
            verified=Path(tmp)/"verified.sqlite"
            with partial.open("rb") as src,verified.open("wb") as out:
                zstd.ZstdDecompressor().copy_stream(src,out)
            def digest(path):
                h=hashlib.sha256()
                with path.open("rb") as file:
                    for chunk in iter(lambda:file.read(1024*1024),b""):
                        h.update(chunk)
                return h.digest()
            if digest(snapshot)!=digest(verified):
                raise IOError("Copia de calidad incompleta.")
            if target.exists():
                shutil.copyfile(target,target.with_suffix(".previous.zst"))
            os.replace(partial,target)

    def page(self,memory,split,start_seq,max_games):
        # fingerprint valida la identidad aunque cambie la numeración del corpus.
        rows=self.conn.execute("SELECT fingerprint,labels FROM reviewed WHERE split=? AND seq>? ORDER BY seq",(split,start_seq))
        result=[]
        for key,labels in rows:
            record=memory.conn.execute("SELECT seq,payload FROM corpus WHERE fingerprint=? AND split=? AND seq>?",
                                       (key,split,start_seq)).fetchone()
            if record:
                result.append((record[0],key,record[1],labels))
                if len(result)>=max_games:
                    break
        return result

    def examples(self,memory,split,epoch=0,max_games=2000,start_seq=0):
        if split not in ("train","validation"):
            raise ValueError("El test no se filtra ni se usa para entrenar.")
        for seq,key,payload,labels in self.page(memory,split,start_seq,max_games):
            labels=json.loads(labels)
            # Selección fija revisada: reutiliza etiquetas, no hace búsquedas al entrenar.
            for example in jade_record_examples(payload,epoch,key,100,include_board=split!="train"):
                label=labels.get(str(example["ply"]))
                if label and label["uci"]==example["uci"] and label["weight"]>0:
                    example.update(seq=seq,weight=label["weight"])
                    yield example

    def close(self):
        self.conn.close()


def open_jade_quality(local_dir,model_dir,rhythm):
    path=Path(local_dir)/f"calidad_{rhythm}.sqlite"
    backup=Path(model_dir)/"revision_calidad"
    if not path.exists():
        errors=[]
        for name in (f"calidad_{rhythm}.sqlite.zst",f"calidad_{rhythm}.sqlite.previous.zst"):
            source=backup/name
            if not source.exists():
                continue
            temp=path.with_suffix(".restore");path.parent.mkdir(parents=True,exist_ok=True)
            try:
                with source.open("rb") as src,temp.open("wb") as out:
                    zstd.ZstdDecompressor().copy_stream(src,out)
                check=sqlite3.connect(str(temp))
                try:
                    valid=check.execute("PRAGMA integrity_check").fetchone()[0]=="ok"
                    same=check.execute("SELECT value FROM settings WHERE key='rhythm'").fetchone()==(rhythm,)
                    if not valid or not same:
                        raise ValueError("Copia de revisión incompatible.")
                finally:
                    check.close()
                os.replace(temp,path);break
            except (OSError,ValueError,sqlite3.Error,zstd.ZstdError) as exc:
                errors.append(str(exc));temp.unlink(missing_ok=True)
        if errors and not path.exists():
            raise RuntimeError("No se recuperó la revisión: "+"; ".join(errors))
    return JadeQuality(path,rhythm,backup)


def prepare_jade_quality(memory,quality,binary,train_games=250,validation_games=150,
                         positions_per_game=6,seconds=.12,nodes=20000,log=print):
    """Revisa SOLO una muestra: no vuelve a analizar millones de partidas.

    Cursor de revisión independiente. Evalúa la posición anterior y compara
    jugada humana/referencia en la misma búsqueda MultiPV; si cambia el signo
    de mate o hay pérdidas muy graves, repite con un presupuesto mayor.
    """
    if min(train_games,validation_games,positions_per_game,nodes)<1 or seconds<=0:
        raise ValueError("Límites de revisión positivos.")
    if quality.rhythm!=memory.rhythm:
        raise ValueError("Ritmos distintos.")
    sync_jade_quality(memory,quality)
    engine=chess.engine.SimpleEngine.popen_uci(str(binary),timeout=60)
    engine.configure({"Threads":1,"Hash":64,"Skill Level":20,"UCI_LimitStrength":False})
    engine_name=engine.id.get("name","Stockfish");done=0;last_save=time.monotonic();stats=Counter()
    try:
        for split,target in (("train",train_games),("validation",validation_games)):
            existing=quality.conn.execute("SELECT COUNT(*) FROM reviewed WHERE split=?",(split,)).fetchone()
            needed=target if split=="train" else max(0,target-existing[0])
            if needed==0:
                continue
            cursor_key="sample_cursor_"+split
            saved_cursor=quality.conn.execute("SELECT value FROM settings WHERE key=?",(cursor_key,)).fetchone()
            start_seq=int(saved_cursor[0]) if saved_cursor else 0
            records=memory.conn.execute("SELECT seq,fingerprint,payload FROM corpus WHERE split=? AND seq>? ORDER BY seq",
                                        (split,start_seq))
            added=0
            for seq,key,payload in records:
                if quality.conn.execute("SELECT 1 FROM reviewed WHERE fingerprint=?",(key,)).fetchone():
                    with quality.conn:
                        quality.conn.execute("INSERT OR REPLACE INTO settings VALUES(?,?)",(cursor_key,str(seq)))
                    continue
                labels={}
                for example in jade_record_examples(payload,0,key,positions_per_game,include_board=True):
                    board=example["board"];move=chess.Move.from_uci(example["uci"])
                    token=object()
                    first=engine.analyse(board,chess.engine.Limit(time=seconds,nodes=nodes),game=token)
                    ref=first["pv"][0];best=first["score"].pov(board.turn);played=best
                    if move!=ref:
                        infos=engine.analyse(board,chess.engine.Limit(time=seconds,nodes=nodes),
                                             multipv=2,root_moves=[ref,move],game=token)
                        scores={x["pv"][0]:x["score"].pov(board.turn) for x in infos if x.get("pv") and "score" in x}
                        if len(scores)!=2:
                            continue  # Sin comparación fiable no inventa una etiqueta.
                        best=max(scores.values());played=scores[move]
                        weight,loss,kind=jade_quality_weight(best,played)
                        if weight==0:
                            infos=engine.analyse(board,chess.engine.Limit(time=seconds*3,nodes=nodes*3),
                                                 multipv=2,root_moves=[ref,move],game=token)
                            scores={x["pv"][0]:x["score"].pov(board.turn) for x in infos if x.get("pv") and "score" in x}
                            if len(scores)!=2:
                                continue
                            best=max(scores.values());played=scores[move]
                    weight,loss,kind=jade_quality_weight(best,played)
                    labels[str(example["ply"])]=dict(uci=example["uci"],weight=weight,loss_cp=loss,kind=kind,
                        engine=engine_name,nodes=nodes,seconds=seconds)
                with quality.conn:
                    quality.conn.execute("INSERT OR IGNORE INTO reviewed VALUES(?,?,?,?)",(key,seq,split,json.dumps(labels)))
                    quality.conn.execute("INSERT OR REPLACE INTO settings VALUES(?,?)",(cursor_key,str(seq)))
                for label in labels.values():
                    stats["peso completo" if label["weight"]==1 else "excluidas" if label["weight"]==0 else "peso reducido"]+=1
                done+=1
                added+=1
                if done%10==0:
                    log(f"Revisión: {done:,} partidas nuevas · {split}; avance guardado localmente.")
                if time.monotonic()-last_save>=90:
                    quality.save();last_save=time.monotonic();log("Revisión respaldada en Drive.")
                if added>=needed:
                    break
    except KeyboardInterrupt:
        log("Revisión detenida. Se conservan las partidas completadas.")
    finally:
        try:
            engine.quit()
        finally:
            quality.save()
    report=quality.summary();log("Partidas revisadas: "+str(report))
    log("Posiciones del lote nuevo: "+str(dict(stats)))
    return report


def load_jade_play_policy(folder,rhythm="rapid"):
    corrected=load_jade_policy(Path(folder)/"calidad_22",rhythm)
    return corrected if corrected and corrected.active else load_jade_policy(folder,rhythm)

"""Incorporación revisada: analiza antes de sumar frecuencias, sin tocar test.

El corpus conserva las jugadas originales. Las etiquetas de calidad se confirman
en la MISMA transacción que la partida y el cursor. La caché de análisis parcial
es independiente; nunca basta por sí sola para aportar ejemplos al entrenamiento.
"""


def jade_review_move(engine,board,move,seconds,nodes,token):
    """Comparación desde el jugador al turno, nunca desde el ganador de la partida."""
    first=engine.analyse(board,chess.engine.Limit(time=seconds,nodes=nodes),game=token)
    if not first.get("pv") or "score" not in first:
        raise RuntimeError("Stockfish no devolvió una evaluación completa; vuelve a ejecutar 5.")
    ref=first["pv"][0];best=first["score"].pov(board.turn);played=best
    if move!=ref:
        for multiplier in (1,3):
            infos=engine.analyse(board,chess.engine.Limit(time=seconds*multiplier,nodes=nodes*multiplier),
                                 multipv=2,root_moves=[ref,move],game=token)
            scores={x["pv"][0]:x["score"].pov(board.turn) for x in infos if x.get("pv") and "score" in x}
            if ref not in scores or move not in scores:
                raise RuntimeError("Análisis incompleto; se conserva el avance para reanudar.")
            best=max(scores.values());played=scores[move]
            if jade_quality_weight(best,played)[0]>0:
                break
    weight,loss,kind=jade_quality_weight(best,played)
    return dict(uci=move.uci(),weight=weight,loss_cp=loss,kind=kind,
                engine=engine.id.get("name","Stockfish"),seconds=seconds,nodes=nodes)


def jade_rare_error_weight(label,key,ply,split,error_fraction=.02):
    """Conserva una muestra fija de fallos graves con peso 0,05, SOLO en train.

    2 % de selección no significa 2 % de errores al jugar. No se vuelve a sortear
    en cada época, ni se conservan por este método errores de mate detectados.
    """
    result=dict(label);result["rare_error"]=False
    if split=="train" and label["weight"]==0 and label["kind"]=="cp":
        sample=int.from_bytes(hashlib.sha256(f"jade-error-v1:{key}:{ply}".encode()).digest()[:8],"big")/2**64
        if sample<float(error_fraction):
            result.update(weight=.05,rare_error=True)
    return result


def sync_jade_quality(memory,quality):
    """Restaura etiquetas confirmadas desde la copia principal, por bloques.

    Permite recuperar tras una desconexión entre las dos copias de Drive. El
    fingerprint verifica el cursor si se recupera una memoria anterior.
    """
    if memory.rhythm!=quality.rhythm:
        raise ValueError("Ritmos distintos.")
    present=memory.conn.execute("SELECT 1 FROM sqlite_master WHERE name='quality_labels'").fetchone()
    if not present:
        return 0
    row=quality.conn.execute("SELECT value FROM settings WHERE key='advanced_cursor'").fetchone()
    cursor=json.loads(row[0]) if row else [0,""]
    if cursor[0] and memory.conn.execute("SELECT fingerprint FROM quality_labels WHERE seq=?",(cursor[0],)).fetchone()!=(cursor[1],):
        cursor=[0,""]
    copied=0
    while True:
        rows=memory.conn.execute("SELECT fingerprint,seq,split,labels FROM quality_labels WHERE seq>? ORDER BY seq LIMIT 256",
                                 (cursor[0],)).fetchall()
        if not rows:
            return copied
        with quality.conn:
            quality.conn.executemany("INSERT OR REPLACE INTO reviewed VALUES(?,?,?,?)",rows)
            cursor=[rows[-1][1],rows[-1][0]]
            quality.conn.execute("INSERT OR REPLACE INTO settings VALUES('advanced_cursor',?)",(json.dumps(cursor),))
        copied+=len(rows)


class JadeAdvancedReviewer:
    """Un motor persistente; caché RAM acotada; puntos de recuperación en disco."""
    def __init__(self,quality,binary,seconds=.08,nodes=12000,error_fraction=.02,log=print):
        from collections import OrderedDict
        if not .02<=float(seconds)<=5 or int(nodes)<100 or not 0<=float(error_fraction)<=.05:
            raise ValueError("Usa 0,02–5 s, al menos 100 nodos y 0–5 % de errores conservados.")
        self.quality=quality;self.log=log
        self.config=dict(version="2.2.1",seconds=float(seconds),nodes=int(nodes),error_fraction=float(error_fraction))
        with quality.conn:
            quality.conn.execute('''CREATE TABLE IF NOT EXISTS advanced_review(
                fingerprint TEXT PRIMARY KEY,split TEXT NOT NULL,labels TEXT NOT NULL,
                complete INTEGER NOT NULL,config TEXT NOT NULL)''')
        self.engine=chess.engine.SimpleEngine.popen_uci(str(binary),timeout=60)
        try:
            self.engine.configure({"Threads":min(2,os.cpu_count() or 1),"Hash":64,
                                   "Skill Level":20,"UCI_LimitStrength":False})
        except BaseException:
            self.engine.quit();raise
        self.cache=OrderedDict();self.stats=Counter();self.accepted=0
        self.started=self.last_save=self.last_log=time.monotonic()
        self.last_key=None

    def _persist(self,key,split,labels,complete,config):
        with self.quality.conn:
            self.quality.conn.execute("INSERT OR REPLACE INTO advanced_review VALUES(?,?,?,?,?)",
                (key,split,json.dumps(labels),int(complete),json.dumps(config)))
        if time.monotonic()-self.last_save>=90:
            self.quality.save();self.last_save=time.monotonic()
            self.log("Revisión parcial respaldada; se puede continuar desde aquí.")

    def review(self,record,key,split):
        if split=="test":
            raise ValueError("El test no se revisa para incorporarlo al aprendizaje.")
        if split not in ("train","validation"):
            raise ValueError("Grupo de datos desconocido.")
        self.last_key=key
        row=self.quality.conn.execute("SELECT split,labels,complete,config FROM advanced_review WHERE fingerprint=?",(key,)).fetchone()
        if row and row[0]!=split:
            raise ValueError("La partida cambió de grupo; no se mezclan entrenamiento y examen.")
        labels=json.loads(row[1]) if row else {}
        config=json.loads(row[3]) if row else self.config.copy()
        if row and row[2]:
            return labels
        board=chess.Board();token=object();new=0
        try:
            for ply,code in enumerate(record["moves"]):
                move=unpack_move(code)
                if move not in board.legal_moves:
                    raise ValueError("Secuencia ilegal; no se incorpora.")
                profile=rating_band(record["ratings"][int(board.turn)])
                if profile in PROFILES and str(ply) not in labels:
                    if board.legal_moves.count()==1:
                        label=dict(uci=move.uci(),weight=1.,loss_cp=0,kind="forzada",rare_error=False)
                    else:
                        # Historial incluido: no confunde repeticiones ni la regla de 50 movimientos.
                        cache_key=(tuple(record["moves"][:ply]),code,config["seconds"],config["nodes"])
                        label=self.cache.get(cache_key)
                        if label is None:
                            label=jade_review_move(self.engine,board,move,config["seconds"],config["nodes"],token)
                            self.cache[cache_key]=label
                            if len(self.cache)>2048:
                                self.cache.popitem(last=False)
                        else:
                            self.cache.move_to_end(cache_key)
                        label=jade_rare_error_weight(label,key,ply,split,config["error_fraction"])
                    labels[str(ply)]=dict(label,profile=profile)
                    new+=1
                    self.stats["errores conservados" if label.get("rare_error") else "peso completo" if label["weight"]==1
                               else "excluidas" if label["weight"]==0 else "peso reducido"]+=1
                    if new%10==0:
                        self._persist(key,split,labels,False,config)
                    if time.monotonic()-self.last_log>=10:
                        self.log(f"Analizando partida {self.accepted+1}: {ply+1}/{len(record['moves'])} medias jugadas · {split}.")
                        self.last_log=time.monotonic()
                board.push(move)
            self._persist(key,split,labels,True,config)
            return labels
        except BaseException:
            # Un movimiento a medio analizar se repite; los terminados se reutilizan.
            self._persist(key,split,labels,False,config)
            raise

    def after_commit(self,memory,delta):
        if delta.get("nueva"):
            self.accepted+=delta["nueva"]
            sync_jade_quality(memory,self.quality)
            if self.last_key and memory.conn.execute("SELECT 1 FROM quality_labels WHERE fingerprint=?",(self.last_key,)).fetchone():
                with self.quality.conn:
                    self.quality.conn.execute("DELETE FROM advanced_review WHERE fingerprint=?",(self.last_key,))
                self.last_key=None
            if self.accepted%5==0:
                elapsed=(time.monotonic()-self.started)/60
                self.log(f"Avanzado: {self.accepted} partidas incorporadas · {elapsed:.1f} min · "+str(dict(self.stats)))

    def finish(self,memory):
        sync_jade_quality(memory,self.quality)
        self.quality.save()
        self.log("Revisión guardada. Para ajustar la red: celda 9 con USAR_REVISION_CALIDAD=True.")
        self.log("Partidas revisadas disponibles: "+str(self.quality.summary()))
        self.log("Posiciones analizadas en esta sesión: "+str(dict(self.stats)))

    def close(self):
        try:
            self.engine.quit()
        finally:
            self.quality.save()

"""Informes legibles de predicción humana, independientes del motor Stockfish."""

def jade_metrics_html(metrics,previous=None,title="Examen de Jade"):
    import html
    esc=lambda value:html.escape(str(value))
    def percent(value):
        return f"{100*float(value):.2f}".replace(".",",")+" %"
    def number(value):
        return f"{int(value):,}".replace(",",".")
    n=int(metrics.get("n",0))
    css='''
    :root{color-scheme:light}*{box-sizing:border-box}body{margin:0;padding:18px;background:#f4f7f5;color:#20352c;font:15px/1.5 system-ui,sans-serif}
    main{max-width:920px;margin:auto}h1{font-size:25px;margin:0 0 6px}h2{font-size:19px;margin-top:26px}p{margin:8px 0 14px}
    .cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px}.card,.note{background:white;border:1px solid #d4e2d9;border-radius:12px;padding:16px}
    .value{display:block;font-size:32px;font-weight:750;color:#126442}.small{font-size:13px;color:#53655b}.notice{background:#e3f0e9;border-left:4px solid #26714d;padding:12px;border-radius:6px}
    .table{overflow-x:auto;background:white;border:1px solid #d4e2d9;border-radius:10px}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
    th,td{padding:10px;text-align:right;border-bottom:1px solid #e5ede8;white-space:nowrap}th:first-child,td:first-child{text-align:left}th{background:#eaf2ed}
    .plus{color:#14643f;font-weight:650}.minus{color:#954522;font-weight:650}details{margin-top:16px}summary{cursor:pointer;font-weight:650}li{margin:8px 0}
    @media(max-width:600px){.table table,.table tbody{display:block}.table thead{display:none}.table tr{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));border-bottom:2px solid #d4e2d9}.table td{display:block;text-align:left;white-space:normal;border:0}.table td:first-child{grid-column:1/-1;background:#eaf2ed;font-weight:700}.table td:not(:first-child)::before{content:attr(data-label);display:block;color:#53655b;font-size:12px;font-weight:400}}
    @media(max-width:420px){body{padding:10px}.cards{grid-template-columns:1fr}.value{font-size:28px}th,td{padding:8px}}
    '''
    start=f'<!doctype html><html lang="es"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{esc(title)}</title><style>{css}</style><main><h1>{esc(title)}</h1>'
    if not n:
        return start+'<p class="notice">Todavía no hay posiciones evaluables. Añade partidas nuevas con la celda 5. Las frecuencias antiguas no sirven como examen independiente.</p></main></html>'
    top1=float(metrics["top1"]);baseline=float(metrics["baseline_top1"])
    gain=(top1-baseline)*100
    verdict="El predictor supera la referencia en acierto y error medio." if top1>baseline and metrics["nll"]<metrics["baseline_nll"] else "Resultado mixto: revisa el acierto y el error medio por separado."
    if top1<=baseline and metrics["nll"]>=metrics["baseline_nll"]:
        verdict="El predictor todavía no supera la referencia en este examen."
    if n<200:
        verdict="Muestra pequeña: interpreta estos resultados con cautela. "+verdict
    summary=[f'{number(n)} posiciones']
    if "games_with_examples" in metrics:
        summary.append(f'{number(metrics["games_with_examples"])} partidas con ejemplos')
    if "seconds" in metrics:
        summary.append(f'{esc(metrics["seconds"])} s')
    parts=[start,'<p>'+" · ".join(summary)+'</p>',
           f'<p class="notice">{esc(verdict)} Esto mide imitación humana, no ELO ni fuerza de juego.</p>',
           '<div class="cards">',
           f'<div class="card">Jugada exacta<span class="value">{percent(top1)}</span><span class="small">Aproximadamente {round(top1*100)} de cada 100 jugadas humanas.</span></div>',
           f'<div class="card">Entre sus tres favoritas<span class="value">{percent(metrics["top3"])}</span><span class="small">La jugada real está entre las tres primeras opciones.</span></div>',
           f'<div class="card">Ventaja sobre frecuencias<span class="value">{gain:+.2f} pp</span><span class="small">Referencia: {percent(baseline)}. «pp» significa puntos porcentuales.</span></div></div>']
    if metrics.get("objective")=="human_quality_weighted":
        parts.insert(2,'<p class="notice">Validación revisada por Stockfish: porcentajes y error ponderados por calidad. Se excluyen las etiquetas de peso cero. No compares estos números directamente con el test humano original. Mejorar aquí tampoco garantiza fuerza de juego.</p>')
    same=bool(previous and metrics.get("selection_signature") and previous.get("selection_signature")==metrics["selection_signature"])
    if same:
        diff=(top1-previous["top1"])*100
        parts.append(f'<h2>Respecto al examen anterior</h2><p>Mismas posiciones: acierto de {percent(previous["top1"])} a <strong>{percent(top1)}</strong> ({diff:+.2f} puntos). Error de {previous["nll"]:.3f} a {metrics["nll"]:.3f}.</p>')
        if previous.get("reference_signature")!=metrics.get("reference_signature"):
            parts.append('<p class="small">La memoria de frecuencias ha cambiado. Su referencia anterior no es idéntica.</p>')
    elif previous:
        parts.append('<p class="small">No se calcula una mejora frente al informe anterior: cambió la selección o aquel informe no guardaba una huella para comprobarla.</p>')
    parts.append('<h2>Dónde acierta</h2><p class="small">«Red» es el predictor por sí solo. «Frecuencias» usa las jugadas registradas. Esta tabla no evalúa la mezcla completa con Stockfish.</p>')
    groups=metrics.get("groups",{})
    def table(keys):
        body=['<div class="table"><table><thead><tr><th>Grupo</th><th>Posiciones</th><th>Red</th><th>Frecuencias</th><th>Diferencia</th><th>Entre 3</th></tr></thead><tbody>']
        names={"apertura":"Apertura","medio/final":"Medio juego / final","conocida":"Posición conocida","nueva":"Posición nueva"}
        for key in keys:
            if key not in groups:
                continue
            row=groups[key];delta=100*(row["top1"]-row["baseline_top1"])
            label=names.get(key,"Perfil "+key)+( " · muestra pequeña" if row["n"]<30 else "")
            body.append(f'<tr><td>{esc(label)}</td><td data-label="Posiciones">{number(row["n"])}</td><td data-label="Acierto de la red">{percent(row["top1"])}</td><td data-label="Frecuencias">{percent(row["baseline_top1"])}</td><td data-label="Diferencia" class="{"plus" if delta>=0 else "minus"}">{delta:+.2f} pp</td><td data-label="Entre sus tres favoritas">{percent(row["top3"])}</td></tr>')
        return ''.join(body)+"</tbody></table></div>"
    parts.extend([table(["apertura","medio/final","conocida","nueva"]),'<h2>Por perfil de rating</h2>',table([str(p) for p in PROFILES]),
                  '<p class="small">Los grupos se solapan: una posición puede ser de apertura, nueva y del perfil 1300. No sumes estas tablas entre sí. Los perfiles no son ELO certificados de Jade.</p>',
                  '<h2>Cómo interpretarlo</h2><ul><li>Equivocarse al adivinar no implica una mala jugada: una persona puede elegir entre varias opciones razonables.</li><li>Las posiciones conocidas pueden favorecer a las frecuencias. Conservamos ambas fuentes de información.</li><li>Usa la validación para ajustar el entrenamiento y reserva el test para revisiones puntuales.</li></ul>',
                  f'<details><summary>Detalle técnico: error de predicción</summary><p>Red: <strong>{metrics["nll"]:.3f}</strong> · Frecuencias: <strong>{metrics["baseline_nll"]:.3f}</strong>. Menor es mejor. Penaliza asignar poca probabilidad a la jugada que realmente se hizo; no es un porcentaje.</p></details>',
                  '<p class="small">Se excluyen las posiciones con una sola jugada legal. Las partidas reservadas no entrenan la red; pueden compartir aperturas o jugadores con el entrenamiento.</p></main></html>'])
    return ''.join(parts)


def mostrar_metricas_jade(metrics,folder=None,previous=None,title="Examen de Jade"):
    """Muestra un informe móvil y guarda HTML + JSON cuando se indica carpeta."""
    import html
    from IPython.display import HTML,display
    document=jade_metrics_html(metrics,previous,title)
    result={}
    if folder is not None:
        folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
        stem=metrics.get("split","examen")+"_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%f")
        page=folder/(stem+".html");page.write_text(document,encoding="utf-8")
        data=folder/(stem+".json");jade_atomic_json(data,metrics)
        result=dict(html=str(page),json=str(data))
    display(HTML('<iframe title="Resultados de Jade" sandbox srcdoc="'+html.escape(document,quote=True)+'" style="width:100%;height:1120px;border:0;border-radius:12px"></iframe>'))
    if result:
        print("Informe HTML y datos JSON guardados en:",folder)
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

"""Interfaz Jade 2.0: tablero SVG, reloj de práctica y controles táctiles.

No cambia la base de frecuencias ni entrena con las jugadas del usuario.
El constructor también funciona sin Colab para verificar la lógica y el HTML.
"""
import html
import json
import math
import os
import tempfile
import time
import uuid
from pathlib import Path

import chess
import chess.pgn
import chess.svg
from IPython.display import HTML, JSON, display


class JadeBoard:
    VERSION = "2.2.1"
    PROFILES = (1200, 1300, 1400, 1500, 1600)

    def __init__(self, jade, game_path=None, previous=None, now=None, sleeper=None):
        self.jade = jade
        self.board = chess.Board()
        self.human, self.orientation = chess.WHITE, chess.WHITE
        self.profile, self.beta = 1300, 1.0
        self.selected, self.pending = None, []
        self.finished, self.busy, self.closed, self.started = False, False, False, False
        self.result, self.message = "*", "Elige tu color y pulsa Jugar."
        self.last_details = None
        self.now, self.sleep = now or time.monotonic, sleeper or time.sleep
        self.initial_seconds, self.increment = 600., 0.
        self.remaining = [600., 600.]  # Índices: negras=0, blancas=1.
        self.paused, self.clock_started = False, None
        self.clock_history, self.move_clocks = [], []
        self.maximum, self.pace = 15, 1.
        self.revision = 0
        self.callback_name = "jade.ui." + uuid.uuid4().hex
        self.game_path = Path(game_path) if game_path is not None else None
        self.save_note = "La partida solo se conserva en esta sesión."
        self.load_error = None
        if previous is not None:
            if hasattr(previous, "settle_clock"):
                previous.settle_clock()
            # Conserva el historial, también al actualizar desde el tablero 1.0.
            self.board = previous.board.copy(stack=True)
            self.human = bool(previous.human)
            self.orientation = getattr(previous, "orientation", self.human)
            self.profile, self.beta = previous.profile, previous.beta
            self.finished, self.result = previous.finished, previous.result
            self.last_details = previous.last_details
            self.started = getattr(previous, "started", True)
            if hasattr(previous, "initial_seconds"):
                self.initial_seconds, self.increment = previous.initial_seconds, previous.increment
                self.remaining = list(previous.remaining)
                self.clock_history = [list(x) for x in previous.clock_history]
                self.move_clocks = list(previous.move_clocks)
                self.maximum, self.pace = previous.maximum, previous.pace
                self.paused = bool(self.started and not self.finished)
            else:
                # A una partida de 1.0/1.1 no se le inventa tiempo ya consumido.
                self.initial_seconds, self.remaining = 0., [0., 0.]
                self.clock_history = [[0., 0.] for _ in self.board.move_stack]
                self.move_clocks = [None for _ in self.board.move_stack]
            self.load_error = getattr(previous, "load_error", None)
            if self.load_error:
                self.message = self.load_error
                self.save_note = "El archivo original se conserva."
            else:
                self.message = "Partida conservada. Pulsa Continuar reloj." if self.paused else "Interfaz actualizada. Tu partida se conserva."
                self.save_game()
        elif self.game_path is not None and self.game_path.exists():
            try:
                self.restore_game(json.loads(self.game_path.read_text(encoding="utf-8")))
                self.message = "Partida recuperada en pausa. Pulsa Continuar reloj." if self.paused else "Partida recuperada. Puedes continuar."
                self.save_note = "Partida recuperada de la copia guardada."
            except (OSError, ValueError, KeyError, TypeError) as exc:
                self.load_error = "No se pudo recuperar la partida: " + str(exc)
                self.message = self.load_error
                self.save_note = "El archivo original se conserva."

    def restore_game(self, data):
        if data["schema"] not in (1, 2) or data["rhythm"] != self.jade.memory.rhythm:
            raise ValueError("La copia tiene otro formato o ritmo.")
        board = chess.Board(data["root_fen"])
        if not board.is_valid():
            raise ValueError("Posición inicial no válida.")
        for uci in data["moves"]:
            board.push_uci(uci)
        profile, beta = int(data["profile"]), float(data["beta"])
        if profile not in self.PROFILES or not math.isfinite(beta) or not 0 <= beta <= 2:
            raise ValueError("Ajustes de partida no válidos.")
        if data["result"] not in ("*", "1-0", "0-1", "1/2-1/2"):
            raise ValueError("Resultado no válido.")
        self.board, self.profile, self.beta = board, profile, beta
        self.human = bool(data["human"])
        self.orientation = bool(data.get("orientation", self.human))
        self.started, self.finished = bool(data["started"]), bool(data["finished"])
        self.result = data["result"]
        self.last_details = data.get("last_details")
        clock = data.get("clock", {})
        self.initial_seconds = float(clock.get("initial", 0))
        self.increment = float(clock.get("increment", 0))
        self.remaining = [float(x) for x in clock.get("remaining", [0., 0.])]
        if len(self.remaining) != 2 or any(not math.isfinite(x) or x < 0 for x in self.remaining):
            raise ValueError("Reloj guardado no válido.")
        self.clock_history = clock.get("history", [[0., 0.] for _ in board.move_stack])
        self.move_clocks = clock.get("moves", [None for _ in board.move_stack])
        if len(self.clock_history) != len(board.move_stack) or len(self.move_clocks) != len(board.move_stack):
            raise ValueError("Historial del reloj incompleto.")
        self.maximum, self.pace = data.get("maximum", 15), float(data.get("pace", 1.))
        if self.maximum not in (12, 15) or self.pace not in (.5, 1., 1.5):
            raise ValueError("Configuración del selector no válida.")
        self.paused = bool(self.initial_seconds and self.started and not self.finished)
        self.clock_started = None
        self.jade.new_game()
        self.update_outcome()

    def game_data(self):
        return dict(schema=2, rhythm=self.jade.memory.rhythm,
                    root_fen=self.board.root().fen(),
                    moves=[move.uci() for move in self.board.move_stack],
                    human=self.human, orientation=self.orientation, profile=self.profile,
                    beta=self.beta, started=self.started, finished=self.finished,
                    result=self.result, last_details=self.last_details,
                    maximum=self.maximum, pace=self.pace,
                    clock=dict(initial=self.initial_seconds, increment=self.increment,
                               remaining=self.remaining, history=self.clock_history, moves=self.move_clocks))

    def clock_running(self):
        return bool(self.initial_seconds and self.started and not self.finished and not self.paused and not self.closed)

    def settle_clock(self):
        if not self.clock_running() or self.clock_started is None:
            return
        stamp = self.now()
        side = int(self.board.turn)
        self.remaining[side] = max(0., self.remaining[side] - max(0., stamp - self.clock_started))
        self.clock_started = stamp
        if self.remaining[side] <= 0:
            opponent = not self.board.turn
            self.finished = True
            self.result = "1/2-1/2" if self.board.has_insufficient_material(opponent) else ("1-0" if opponent else "0-1")
            self.message = ("Tu tiempo se ha agotado" if self.board.turn == self.human else "Jade se ha quedado sin tiempo") + " · " + self.result
            self.clock_started = None
            self.pending = []
            self.save_game()

    def clock_view(self):
        values = list(self.remaining)
        if self.clock_running() and self.clock_started is not None:
            side = int(self.board.turn)
            values[side] = max(0., values[side] - max(0., self.now() - self.clock_started))
        return values

    def commit_move(self, move):
        self.settle_clock()
        if self.finished:
            return False
        self.clock_history.append(list(self.remaining))
        mover = int(self.board.turn)
        if self.initial_seconds:
            self.remaining[mover] += self.increment
        self.move_clocks.append(self.remaining[mover] if self.initial_seconds else None)
        self.board.push(move)
        self.clock_started = self.now() if self.clock_running() else None
        return True

    def save_game(self):
        if self.game_path is None:
            return
        temporary = None
        try:
            self.game_path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".part",
                                             dir=self.game_path.parent, delete=False) as out:
                temporary = Path(out.name)
                json.dump(self.game_data(), out, ensure_ascii=False)
                out.flush()
                os.fsync(out.fileno())
            os.replace(temporary, self.game_path)
            place = "Drive" if str(self.game_path).startswith("/content/drive/") else "el almacenamiento configurado"
            self.save_note = "Partida guardada en " + place + "."
        except OSError as exc:
            self.save_note = "No se pudo guardar la partida. Descarga el PGN antes de salir. " + str(exc)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def archive_game(self):
        """Al empezar otra, conserva la anterior como PGN si tiene movimientos."""
        if self.game_path is not None and self.board.move_stack:
            path = self.game_path.parent / ("partida_" + uuid.uuid4().hex + ".pgn")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(self.pgn(), encoding="utf-8")

    def update_outcome(self):
        outcome = self.board.outcome()
        if outcome is not None:
            self.finished, self.result = True, outcome.result()
            reason = {"CHECKMATE": "Jaque mate", "STALEMATE": "Tablas por ahogado",
                      "INSUFFICIENT_MATERIAL": "Tablas por material insuficiente",
                      "SEVENTYFIVE_MOVES": "Tablas por 75 movimientos",
                      "FIVEFOLD_REPETITION": "Tablas por repetición"}
            self.message = reason.get(outcome.termination.name, "Partida terminada") + " · " + self.result

    def pgn(self):
        game = chess.pgn.Game.from_board(self.board)
        game.headers.update({"Event": "Práctica con Jade", "White": "Tú" if self.human else "Jade",
                             "Black": "Jade" if self.human else "Tú", "Result": self.result,
                             "JadeProfile": str(self.profile),
                             "TimeControl": f"{int(self.initial_seconds)}+{int(self.increment)}" if self.initial_seconds else "-"})
        for node, clock in zip(game.mainline(), self.move_clocks):
            if clock is not None:
                node.set_clock(clock)
        return str(game)

    def svg(self):
        targets = {move.to_square for move in self.board.legal_moves if move.from_square == self.selected}
        svg = chess.svg.board(
            self.board, orientation=self.orientation, coordinates=False,
            lastmove=self.board.peek() if self.board.move_stack else None,
            check=self.board.king(self.board.turn) if self.board.is_check() else None,
            fill={self.selected: "#f5cc69"} if self.selected is not None else {},
            colors={"square light": "#edf1df", "square dark": "#739889",
                    "square light lastmove": "#d9dc8a", "square dark lastmove": "#b7c46c"})
        # El SVG sin coordenadas usa casillas de 45 unidades. Las piezas proceden
        # de python-chess; estas marcas solo indican los destinos legales.
        dots = []
        for square in targets:
            file, rank = chess.square_file(square), chess.square_rank(square)
            col = file if self.orientation else 7 - file
            row = 7 - rank if self.orientation else rank
            x, y = 45 * col + 22.5, 45 * row + 22.5
            if self.board.piece_at(square):
                dots.append(f'<circle cx="{x}" cy="{y}" r="19" fill="none" stroke="#153f32" stroke-width="3" opacity=".7"/>')
            else:
                dots.append(f'<circle cx="{x}" cy="{y}" r="6" fill="#153f32" opacity=".45"/>')
        return svg.replace("</svg>", "".join(dots) + "</svg>")

    def state(self):
        self.update_outcome()
        files = list("abcdefgh") if self.orientation else list("hgfedcba")
        ranks = list(range(8, 0, -1)) if self.orientation else list(range(1, 9))
        names = {chess.PAWN: "peón", chess.KNIGHT: "caballo", chess.BISHOP: "alfil",
                 chess.ROOK: "torre", chess.QUEEN: "dama", chess.KING: "rey"}
        targets = {m.to_square for m in self.board.legal_moves if m.from_square == self.selected}
        squares = []
        for rank in ranks:
            for file in files:
                name = file + str(rank)
                sq = chess.parse_square(name)
                piece = self.board.piece_at(sq)
                label = name + (": " + names[piece.piece_type] + (" blanco" if piece.color else " negro") if piece else ": vacía")
                squares.append(dict(name=name, label=label, selected=sq == self.selected, target=sq in targets))
        replay, history = self.board.root(), []
        for move in self.board.move_stack:
            number, white = replay.fullmove_number, replay.turn
            san = replay.san(move)
            if white or not history:
                history.append([number, san if white else "…", ""])
            if not white:
                history[-1][2] = san
            replay.push(move)
        ai_pending = self.started and not self.finished and not self.paused and self.board.turn != self.human and not self.closed
        return dict(revision=self.revision, svg=self.svg(), squares=squares, files=files, ranks=ranks,
                    message=self.message, started=self.started, finished=self.finished, closed=self.closed,
                    human=self.human, orientation=self.orientation, profile=self.profile, beta=self.beta, rhythm=self.jade.memory.rhythm,
                    result=self.result, history=history, pgn=self.pgn(), plies=len(self.board.move_stack),
                    can_move=self.started and not self.finished and not self.closed and not self.paused and self.board.turn == self.human,
                    can_undo=bool(self.board.move_stack) and not (not self.human and len(self.board.move_stack) == 1),
                    can_draw=self.started and not self.finished and self.board.turn == self.human and self.board.can_claim_draw(),
                    ai_pending=ai_pending, promotion=bool(self.pending), save_note=self.save_note,
                    observations=(self.last_details or {}).get("candidate_observations", (self.last_details or {}).get("top5_observations")),
                    decision=self.last_details, maximum=self.maximum, pace=self.pace,
                    clock=dict(enabled=bool(self.initial_seconds), remaining=self.clock_view(),
                               active=int(self.board.turn) if self.clock_running() else None,
                               paused=self.paused, initial=self.initial_seconds, increment=self.increment),
                    turn=("Partida terminada" if self.finished else "En pausa" if self.paused else "Tu turno" if self.board.turn == self.human else "Turno de Jade"))

    def push_human(self, move):
        if move not in self.board.legal_moves:
            self.message = "Esa jugada no es legal. Selecciona un destino marcado."
            return
        self.message = "Has jugado " + self.board.san(move) + "."
        if not self.commit_move(move):
            return
        self.selected, self.pending = None, []
        self.update_outcome()
        self.save_game()  # Se guarda también antes de que Stockfish responda.

    def ai_turn(self):
        if not self.started or self.finished or self.paused or self.board.turn == self.human:
            return
        if self.board.can_claim_draw():
            self.finished, self.result, self.message = True, "1/2-1/2", "Jade reclama tablas."
        else:
            try:
                began = self.now()
                seconds = self.clock_view()[int(self.board.turn)] if self.initial_seconds else None
                move, details = self.jade.choose(self.board, self.profile, self.beta,
                                                 remaining=seconds, maximum=self.maximum, pace=self.pace)
                if move not in self.board.legal_moves:
                    raise ValueError("El motor devolvió una jugada no válida.")
                delay = max(0., details.get("reaction_seconds", 0.) - (self.now() - began))
                if self.initial_seconds:
                    delay = min(delay, max(0., self.clock_view()[int(self.board.turn)] * .5))
                if delay:
                    self.sleep(delay)
                self.settle_clock()
                if not self.finished:
                    self.message = "Jade juega " + self.board.san(move) + ". Te toca."
                    if self.commit_move(move):
                        details["elapsed_seconds"] = self.now() - began
                        self.last_details = details
                        self.update_outcome()
            except Exception as exc:
                self.message = "No se pudo calcular la respuesta. Pulsa Reintentar. Detalle: " + str(exc)
        self.save_game()

    def handle(self, action, payload=None, revision=None):
        """Un único punto de entrada. Python valida todas las jugadas y el turno."""
        if self.closed:
            return dict(self.state(), message="Este tablero se ha cerrado. Usa la salida más reciente de la celda 6.")
        if self.busy:
            return self.state()
        if revision is not None and int(revision) != self.revision:
            return dict(self.state(), message="Tablero sincronizado. Repite tu acción.")
        payload = payload or {}
        self.busy = True
        try:
            self.settle_clock()
            if action == "new":
                profile, beta = int(payload.get("profile", 1300)), float(payload.get("beta", 1))
                if profile not in self.PROFILES or not math.isfinite(beta) or not 0 <= beta <= 2:
                    raise ValueError("Elige uno de los perfiles disponibles y un peso entre 0 y 2.")
                initial, increment = map(int, str(payload.get("time_control", "600+0")).split("+"))
                maximum, pace = int(payload.get("maximum", 15)), float(payload.get("pace", 1.))
                if (initial, increment) not in ((0, 0), (180, 2), (300, 0), (600, 0), (600, 5), (900, 10)) or maximum not in (12, 15) or pace not in (.5, 1., 1.5):
                    raise ValueError("Selecciona un reloj y un ritmo de respuesta válidos.")
                # Un archivo ilegible se conserva antes de crear la nueva partida.
                if self.load_error and self.game_path and self.game_path.exists():
                    recovered = self.game_path.with_name("copia_no_leida_" + uuid.uuid4().hex + ".json")
                    recovered.write_bytes(self.game_path.read_bytes())
                self.archive_game()
                self.jade.new_game()
                self.board = chess.Board()
                self.human = payload.get("color", "white") == "white"
                self.orientation = self.human
                self.profile, self.beta = profile, beta
                self.initial_seconds, self.increment = float(initial), float(increment)
                self.remaining = [float(initial), float(initial)]
                self.clock_history, self.move_clocks = [], []
                self.maximum, self.pace, self.paused = maximum, pace, False
                self.started, self.finished, self.result = True, False, "*"
                self.selected, self.pending, self.last_details = None, [], None
                self.message, self.load_error = "Toca una pieza y después un destino marcado.", None
                self.clock_started = self.now() if initial else None
                self.save_game()
            elif action == "pause" and self.started and not self.finished:
                self.paused = not self.paused
                self.clock_started = self.now() if self.clock_running() else None
                self.message = "Reloj en pausa." if self.paused else "Partida reanudada."
                self.save_game()
            elif action == "tick":
                pass  # settle_clock ya comprueba la caída de bandera.
            elif action == "flip":
                self.orientation = not self.orientation
                if self.started:
                    self.save_game()
            elif action == "undo" and self.board.move_stack:
                if not self.human and len(self.board.move_stack) == 1:
                    self.message = "Todavía no has movido."
                else:
                    while self.board.move_stack:
                        self.board.pop()
                        if self.clock_history:
                            self.remaining = self.clock_history.pop()
                        if self.move_clocks:
                            self.move_clocks.pop()
                        if self.board.turn == self.human:
                            break
                    self.finished, self.result, self.selected = False, "*", None
                    self.pending, self.last_details = [], None
                    self.message = "Turno deshecho. Puedes elegir otra jugada."
                    self.clock_started = self.now() if self.clock_running() else None
                    self.save_game()
            elif action == "resign" and self.started and not self.finished:
                self.finished, self.result = True, "0-1" if self.human else "1-0"
                self.message, self.pending = "Te has rendido. Gana Jade.", []
                self.save_game()
            elif action == "draw" and self.started and not self.finished:
                if self.board.turn == self.human and self.board.can_claim_draw():
                    self.finished, self.result, self.message = True, "1/2-1/2", "Tablas reclamadas."
                    self.pending = []
                    self.save_game()
                else:
                    self.message = "Esta posición todavía no permite reclamar tablas."
            elif action == "ai":
                self.ai_turn()
            elif action == "cancel_promotion":
                self.pending = []
                self.message = "Promoción cancelada. Puedes elegir otro movimiento."
            elif self.started and not self.finished and not self.paused and self.board.turn == self.human:
                if action == "square":
                    sq = chess.parse_square(payload["square"])
                    piece = self.board.piece_at(sq)
                    if piece and piece.color == self.human:
                        self.selected = None if self.selected == sq else sq
                        self.pending = []
                        self.message = "Elige un destino marcado." if self.selected is not None else "Toca una pieza."
                    elif self.selected is not None:
                        moves = [m for m in self.board.legal_moves if m.from_square == self.selected and m.to_square == sq]
                        if any(m.promotion for m in moves):
                            self.pending = moves
                            self.message = "Elige la pieza para promocionar."
                        elif moves:
                            self.push_human(moves[0])
                        else:
                            self.message = "Toca uno de los destinos marcados."
                elif action == "promote":
                    symbol = payload.get("piece", "q")
                    move = next((m for m in self.pending if chess.piece_symbol(m.promotion) == symbol), None)
                    if move is not None:
                        self.push_human(move)
                elif action == "uci":
                    self.push_human(chess.Move.from_uci(str(payload.get("uci", "")).strip().lower()))
        except (ValueError, KeyError, TypeError, OSError) as exc:
            self.message = "No se pudo realizar la acción: " + str(exc)
        finally:
            self.busy = False
            self.revision += 1
        return self.state()

    def callback(self, action, payload=None, revision=None):
        return JSON(self.handle(action, payload, revision))

    def html(self):
        # Evita cerrar el bloque script al incorporar textos del estado.
        initial = json.dumps(self.state(), ensure_ascii=False).replace("<", "\\u003c")
        return (JADE_HTML.replace("__JADE_CALLBACK__", self.callback_name)
                .replace("__JADE_INITIAL__", initial))

    def show(self):
        from google.colab import output
        output.register_callback(self.callback_name, self.callback)
        display(HTML(self.html()))

    def close(self):
        self.settle_clock()
        if self.started:
            self.paused, self.clock_started = True, None
            self.save_game()
        self.closed = True


JADE_HTML = r'''
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="light">
<style>
html,body{margin:0;padding:0;width:100%;min-width:0;background:transparent}
.jade,.jade *{box-sizing:border-box}
.jade{color-scheme:light;width:100%;max-width:460px;min-width:0;margin:0 auto;padding:12px;
  background:#f7f8f3;color:#183a30;border:1px solid #dce4d8;border-radius:18px;
  font:15px/1.45 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;overflow-wrap:anywhere}
.jade button,.jade select,.jade input,.jade textarea{font:inherit;max-width:100%;min-width:0;color:#183a30;color-scheme:light}
.jade button{border:1px solid #c7d4c8;border-radius:10px;background:#fff;padding:10px 12px;min-height:44px;
  cursor:pointer;touch-action:manipulation;-webkit-tap-highlight-color:transparent}
.jade button:disabled{cursor:default;opacity:.5}
.jade button:focus-visible,.jade summary:focus-visible,.jade select:focus-visible{outline:3px solid #216e55;outline-offset:2px}
.jade .brand{display:flex;justify-content:space-between;gap:8px;align-items:center;margin-bottom:12px}
.jade h1{font-size:29px;line-height:1.1;letter-spacing:-1px;margin:0}
.jade .muted{color:#52685b;font-size:12px}
.jade .tag{background:#e4ecdf;padding:5px 9px;border-radius:20px;font-size:12px;white-space:nowrap}
.jade .setup-grid{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin:10px 0}
.jade label{display:block;font-size:13px;font-weight:600}
.jade select,.jade input[type=text],.jade textarea{width:100%;background:#fff;border:1px solid #bbcbbb;border-radius:9px;padding:10px;margin-top:4px;min-height:44px}
.jade .primary{width:100%;background:#1b624b;border-color:#1b624b;color:white;font-weight:650}
.jade details{border-top:1px solid #dce4d8;padding:10px 0}
.jade summary{cursor:pointer;font-weight:600;min-height:32px;display:list-item;margin-left:18px}
.jade .advanced{padding:8px 0}
.jade input[type=range]{width:100%;accent-color:#1b624b}
.jade .status{background:#e9eee3;border-radius:10px;padding:9px 11px;margin:12px 0 8px}
.jade #turn{font-weight:750;display:block}
.jade #message{font-size:13px;display:block;min-height:19px}
.jade .player{display:flex;align-items:center;justify-content:space-between;gap:8px;padding:7px 0;font-size:13px}
.jade .player b{font-weight:650}
.jade .clock{font:700 21px/1.1 ui-monospace,monospace;letter-spacing:1px;padding:7px 10px;border-radius:8px;background:#e4eadf;color:#3d5144;white-space:nowrap}
.jade .clock.active{background:#174d3b;color:#fff}
.jade .clock.low{background:#9d392e;color:#fff}
.jade .clock.paused{opacity:.65}
.jade .board-frame{display:grid;grid-template-columns:12px minmax(0,1fr);grid-template-rows:auto 16px;gap:2px;width:100%;min-width:0}
.jade .ranks{display:grid;grid-template-rows:repeat(8,1fr);font-size:10px;text-align:center;align-items:center;color:#52685b}
.jade .files{grid-column:2;display:grid;grid-template-columns:repeat(8,1fr);font-size:10px;text-align:center;color:#52685b}
.jade .board{position:relative;aspect-ratio:1;width:100%;min-width:0;border-radius:4px;overflow:hidden;background:#739889}
.jade #drawing{position:absolute;inset:0}
.jade #drawing svg{display:block;width:100%;height:100%;max-width:100%}
.jade .squares{position:absolute;inset:0;display:grid;grid-template-columns:repeat(8,minmax(0,1fr));grid-template-rows:repeat(8,minmax(0,1fr))}
.jade .squares button{display:block;width:100%;height:100%;min-height:0;padding:0;border:0;margin:0;border-radius:0;background:transparent!important;opacity:1!important}
.jade .squares button:focus-visible{outline:3px solid #17664b;outline-offset:-3px}
.jade .actions{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin:10px 0}
.jade .actions button{font-size:13px;padding:8px}
.jade .danger{color:#9a3830;border-color:#e5b9b0}
.jade [hidden]{display:none!important}
.jade .promotion,.jade .confirm{background:#fff9e6;border:1px solid #e1c475;border-radius:10px;padding:10px;margin:10px 0}
.jade .promotions{display:grid;grid-template-columns:1fr 1fr;gap:6px;margin:8px 0}
.jade .history{max-height:150px;overflow:auto;background:#fff;border-radius:8px;margin:6px 0}
.jade table{border-collapse:collapse;width:100%;font-size:14px;text-align:left;table-layout:fixed}
.jade th,.jade td{padding:6px 10px;border-bottom:1px solid #e8eee4}
.jade th:first-child,.jade td:first-child{width:40px;color:#617264}
.jade textarea{height:110px;font:12px/1.5 ui-monospace,monospace;resize:vertical}
.jade .footer{margin-top:8px;font-size:11px;color:#607264}
@media(max-width:280px){.jade{padding:8px;border-radius:12px;font-size:14px}.jade .setup-grid{grid-template-columns:1fr}.jade .tag{font-size:10px}.jade button{padding:8px}.jade .actions{gap:5px}}
</style>
<main class="jade" id="jade">
  <header class="brand"><div><h1>Jade</h1><div class="muted">Práctica con protección táctica</div></div><span class="tag">Jade 2.2.1</span></header>
  <details id="setup" open><summary>Preparar partida</summary>
    <div class="setup-grid">
      <label>Perfil de Jade<select id="profile"><option>1200</option><option selected>1300</option><option>1400</option><option>1500</option><option>1600</option></select></label>
      <label>Tú juegas con<select id="color"><option value="white">Blancas</option><option value="black">Negras</option></select></label>
      <label>Reloj<select id="time-control"><option value="600+0">10 minutos</option><option value="600+5">10 min + 5 s</option><option value="900+10">15 min + 10 s</option><option value="300+0">5 minutos</option><option value="180+2">3 min + 2 s</option><option value="0+0">Sin reloj</option></select></label>
      <label>Respuesta de Jade<select id="pace"><option value="1">Natural</option><option value="1.5">Pausada</option><option value="0.5">Ágil</option></select></label>
    </div>
    <details><summary>Ajuste de estilo</summary><div class="advanced">
      <label>Influencia humana: <output id="weight-label">1.00</output><input id="weight" type="range" min="0" max="2" step=".25" value="1"></label>
      <div class="muted">1 es el valor inicial. 0 usa solo el motor y la temperatura. Los perfiles son orientativos, todavía no son ELO medidos.</div>
      <label style="margin-top:8px">Máximo de opciones<select id="maximum"><option>15</option><option>12</option></select></label>
      <div class="muted">Las opciones bajan hasta 3 con poco tiempo. Las jugadas humanas pueden entrar aunque no estén entre las favoritas del motor. El reloj es de práctica; puedes pausarlo.</div>
    </div></details>
    <button id="new" class="primary">Jugar</button>
  </details>
  <div class="confirm" id="confirm" hidden><b id="confirm-text"></b><div class="actions"><button id="confirm-yes">Confirmar</button><button id="confirm-no">Cancelar</button></div></div>
  <div class="status" role="status" aria-live="polite"><span id="turn">Listo para jugar</span><span id="message">Toca Jugar para empezar.</span></div>
  <div class="player"><div><b id="opponent">Jade · perfil 1300</b><div id="rhythm" class="muted">Rapid</div></div><span class="clock" id="clock-top" aria-label="Reloj superior">10:00</span></div>
  <div class="board-frame">
    <div class="ranks" id="ranks" aria-hidden="true"></div>
    <div class="board" id="board"><div id="drawing" aria-hidden="true"></div><div class="squares" id="squares" role="group" aria-label="Tablero de ajedrez"></div></div>
    <div class="files" id="files" aria-hidden="true"></div>
  </div>
  <div class="player"><div><b id="you">Tú · blancas</b><div class="muted">Toca origen y destino</div></div><span class="clock" id="clock-bottom" aria-label="Reloj inferior">10:00</span></div>
  <div class="promotion" id="promotion" role="group" aria-label="Promocionar peón" hidden><b>¿En qué pieza lo conviertes?</b><div class="promotions">
    <button data-piece="q">Dama</button><button data-piece="r">Torre</button><button data-piece="b">Alfil</button><button data-piece="n">Caballo</button>
  </div><button id="cancel-promotion">Cancelar</button></div>
  <button id="retry" class="primary" hidden>Reintentar respuesta de Jade</button>
  <div class="actions"><button id="undo">Deshacer turno</button><button id="flip">Girar tablero</button></div>
  <button id="pause" style="width:100%;margin-bottom:8px">Pausar reloj</button>
  <details id="history-panel"><summary>Movimientos <span id="count"></span></summary><div id="history" class="history"></div></details>
  <details><summary>Opciones de partida</summary>
    <div class="actions"><button id="draw">Reclamar tablas</button><button id="resign" class="danger">Rendirme</button></div>
    <label>Jugada por texto (UCI)<input id="uci" type="text" placeholder="e2e4" autocomplete="off" autocapitalize="none" spellcheck="false"></label>
    <button id="send" style="width:100%;margin-top:6px">Jugar movimiento</button>
    <div class="actions"><button id="download">Descargar PGN</button><button id="copy">Copiar PGN</button></div>
    <textarea id="pgn" readonly aria-label="Partida en PGN"></textarea>
    <div id="evidence" class="muted"></div>
  </details>
  <div class="footer" id="save-note"></div>
  <div class="footer">Piezas: C. M. Burnett · SVG de python-chess.<br>Las partidas jugadas no se añaden a la memoria humana.</div>
</main>
<script>
(()=>{
  const root=document.getElementById('jade');
  const el=id=>root.querySelector('#'+id);
  const callback='__JADE_CALLBACK__';
  let state=__JADE_INITIAL__, waiting=false, confirmation=null, clockStamp=performance.now(), flagSent=false;
  const status=text=>{el('message').textContent=text};
  function fit(){try{google.colab.output.setIframeHeight(Math.ceil(root.getBoundingClientRect().height)+4,true)}catch(e){}}
  function lock(){
    root.querySelectorAll('button,select,input').forEach(x=>x.disabled=waiting||state.closed);
    root.querySelectorAll('.squares button').forEach(x=>x.disabled=waiting||!state.can_move||state.promotion);
    el('undo').disabled=waiting||state.closed||!state.can_undo;
    el('draw').disabled=waiting||state.closed||!state.can_draw;
    el('resign').disabled=waiting||state.closed||!state.started||state.finished;
    el('send').disabled=waiting||!state.can_move||state.promotion;
    el('pause').disabled=waiting||state.closed||!state.started||state.finished;
    el('board').setAttribute('aria-busy',waiting?'true':'false');
  }
  function render(s){
    state=s;clockStamp=performance.now();flagSent=false;
    el('drawing').innerHTML=s.svg;
    el('ranks').replaceChildren(...s.ranks.map(x=>{const e=document.createElement('span');e.textContent=x;return e}));
    el('files').replaceChildren(...s.files.map(x=>{const e=document.createElement('span');e.textContent=x;return e}));
    el('squares').replaceChildren(...s.squares.map(q=>{
      const b=document.createElement('button');b.type='button';b.dataset.square=q.name;
      b.setAttribute('aria-label',q.label+(q.target?', destino legal':''));b.setAttribute('aria-pressed',q.selected?'true':'false');
      b.addEventListener('click',()=>send('square',{square:q.name}));return b;
    }));
    el('turn').textContent=s.closed?'Tablero cerrado':s.started?s.turn:'Listo para jugar';
    status(s.message);
    const humanLabel='Tú · '+(s.human?'blancas':'negras'), jadeLabel='Jade · perfil '+s.profile;
    el('opponent').textContent=s.orientation===s.human?jadeLabel:humanLabel;
    el('rhythm').textContent=s.rhythm==='rapid'?'Datos de rápidas':'Datos de blitz';
    el('you').textContent=s.orientation===s.human?humanLabel:jadeLabel;
    el('pause').textContent=s.clock.paused?'Continuar reloj':'Pausar reloj';
    el('new').textContent=s.started?'Nueva partida':'Jugar';
    el('retry').hidden=!s.ai_pending;
    el('promotion').hidden=!s.promotion;
    el('count').textContent=s.history.length?'('+s.history.length+')':'';
    const table=document.createElement('table');
    const header=document.createElement('tr');
    ['N.º','Blancas','Negras'].forEach(t=>{const th=document.createElement('th');th.textContent=t;header.append(th)});
    table.append(header);
    s.history.forEach(row=>{const tr=document.createElement('tr');row.forEach(t=>{const td=document.createElement('td');td.textContent=t;tr.append(td)});table.append(tr)});
    el('history').replaceChildren(table);el('history').scrollTop=el('history').scrollHeight;
    el('pgn').value=s.pgn;
    el('save-note').textContent=s.save_note;
    const d=s.decision;
    el('evidence').textContent=d?.candidate_count?
      d.candidate_count+' opciones · '+Math.round(100*d.human_mix)+' % de mezcla humana · '+d.candidate_observations+' ejemplos entre esas opciones. Última respuesta: '+(d.elapsed_seconds||0).toFixed(1)+' s.'+
      (d.selected_human_count?' La jugada elegida aparece '+d.selected_human_count+' veces en este perfil.':'')+
      (d.neural_active?' Predictor humano activo.':''):
      s.observations===null?'':s.observations+' observaciones humanas en la última respuesta.';
    lock();clocks();fit();
  }
  function clocks(){
    const c=state.clock;if(!c)return;
    const values=[...c.remaining];
    if(c.active!==null)values[c.active]=Math.max(0,values[c.active]-(performance.now()-clockStamp)/1000);
    const bottom=state.orientation?1:0;
    [['clock-top',1-bottom],['clock-bottom',bottom]].forEach(([id,side])=>{
      const t=values[side];el(id).textContent=c.enabled?(t<10?t.toFixed(1):Math.floor(t/60)+':'+String(Math.floor(t%60)).padStart(2,'0')):'Sin reloj';
      el(id).className='clock'+(c.active===side?' active':'')+(c.enabled&&t<=10?' low':'')+(c.paused?' paused':'');
    });
    if(c.active!==null&&values[c.active]<=0&&!waiting&&!flagSent&&!state.finished){flagSent=true;send('tick');}
  }
  async function call(action,payload={}){
    const result=await google.colab.kernel.invokeFunction(callback,[action,payload,state.revision],{});
    let data=result.data?.['application/json'];
    if(typeof data==='string')data=JSON.parse(data);
    if(!data||!data.squares)throw new Error('No se recibió el estado del tablero.');
    render(data);return data;
  }
  async function send(action,payload={}){
    if(waiting||state.closed)return;
    waiting=true;lock();
    try{
      if(action==='ai'){el('turn').textContent='Jade está pensando…';status('Calculando su respuesta.');el('retry').hidden=true;}
      const s=await call(action,payload);
      if(action==='new'){el('setup').open=false;fit()}
      if(s.ai_pending&&['new','square','promote','uci','pause'].includes(action)){
        el('turn').textContent='Jade está pensando…';status('Calculando su respuesta.');el('retry').hidden=true;
        // Permite pintar tu movimiento antes de iniciar la búsqueda del motor.
        await new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)));
        await call('ai');
      }
    }catch(err){
      status('No se pudo conectar. Comprueba «Conectar» en Colab y vuelve a abrir la celda 6. '+err.message);
    }finally{waiting=false;lock();fit()}
  }
  function confirm(text,action){confirmation=action;el('confirm-text').textContent=text;el('confirm').hidden=false;fit()}
  el('confirm-no').onclick=()=>{confirmation=null;el('confirm').hidden=true;fit()};
  el('confirm-yes').onclick=()=>{const fn=confirmation;confirmation=null;el('confirm').hidden=true;if(fn)fn()};
  el('new').onclick=()=>{
    const start=()=>send('new',{profile:Number(el('profile').value),color:el('color').value,beta:Number(el('weight').value),time_control:el('time-control').value,maximum:Number(el('maximum').value),pace:Number(el('pace').value)});
    if(state.plies&&!state.finished)confirm('¿Dejar esta partida y empezar otra?',start);else start();
  };
  el('weight').oninput=()=>el('weight-label').value=Number(el('weight').value).toFixed(2);
  el('undo').onclick=()=>send('undo');el('flip').onclick=()=>send('flip');el('draw').onclick=()=>send('draw');
  el('resign').onclick=()=>confirm('¿Quieres rendirte en esta partida?',()=>send('resign'));
  el('retry').onclick=()=>send('ai');
  el('pause').onclick=()=>send('pause');
  el('send').onclick=()=>{const uci=el('uci').value;el('uci').value='';send('uci',{uci})};
  el('uci').onkeydown=e=>{if(e.key==='Enter'){e.preventDefault();el('send').click()}};
  root.querySelectorAll('[data-piece]').forEach(b=>b.onclick=()=>send('promote',{piece:b.dataset.piece}));
  el('cancel-promotion').onclick=()=>send('cancel_promotion');
  el('download').onclick=()=>{
    const url=URL.createObjectURL(new Blob([state.pgn],{type:'application/x-chess-pgn'}));
    const a=document.createElement('a');a.href=url;a.download='Jade_partida.pgn';document.body.append(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),10000);
  };
  el('copy').onclick=async()=>{
    try{await navigator.clipboard.writeText(state.pgn);status('PGN copiado.');}
    catch(e){el('pgn').focus();el('pgn').select();status('Texto seleccionado. Mantén pulsado y elige Copiar.');}
  };
  root.querySelectorAll('details').forEach(d=>d.addEventListener('toggle',fit));
  el('profile').value=String(state.profile);el('color').value=state.human?'white':'black';
  el('weight').value=state.beta;el('weight-label').value=Number(state.beta).toFixed(2);
  el('maximum').value=state.maximum;el('pace').value=state.pace;
  el('time-control').value=state.clock.initial+'+'+state.clock.increment;
  el('setup').open=!state.started;render(state);
  const ticker=setInterval(()=>{if(!root.isConnected)clearInterval(ticker);else clocks()},200);
  if(typeof ResizeObserver!=='undefined')new ResizeObserver(fit).observe(root);
  window.addEventListener('resize',fit);
})();
</script>
'''

"""Reanálisis de PGN con Stockfish sin humanización. Exporta HTML, PGN y CSV."""
import csv
import html
import io
import json
import os
import time
import uuid
from pathlib import Path

import chess
import chess.engine
import chess.pgn
import chess.svg
from IPython.display import HTML, display


def jade_score_text(score):
    mate = score.mate()
    if mate is not None:
        return ("#" if score.score(mate_score=100000)>0 else "-#") + str(abs(mate))
    cp = score.score()
    return f"{cp / 100:+.2f}" if cp is not None else "?"


def jade_grade(best, played, same_move, ply):
    """Etiquetas propias, basadas en pérdida de expectativa WDL, no de Lichess."""
    best_mate, played_mate = best.mate(), played.mate()
    loss_cp = None if best.is_mate() or played.is_mate() else max(0, best.score() - played.score())
    loss_wdl = max(0., best.wdl(model="sf", ply=ply).expectation() - played.wdl(model="sf", ply=ply).expectation())
    if same_move:
        return "Mejor del análisis", 0 if loss_cp is not None else None, 0.
    if best_mate is not None and best_mate > 0 and (played_mate is None or played_mate <= 0):
        return "Mate omitido", loss_cp, loss_wdl
    if played_mate is not None and played_mate <= 0 and (best_mate is None or best_mate > 0):
        return "Permite mate", loss_cp, loss_wdl
    if best_mate is not None and played_mate is not None:
        return "Mantiene mate" if played_mate > 0 else "Posición con mate en contra", loss_cp, loss_wdl
    label = ("Precisa" if loss_wdl < .02 else "Buena" if loss_wdl < .05 else
             "Imprecisión" if loss_wdl < .10 else "Error" if loss_wdl < .20 else "Error grave")
    return label, loss_cp, loss_wdl


def jade_report_html(rows, metadata):
    """Informe autónomo. El tablero de cada jugada se prepara antes de exportar."""
    payload = json.dumps(dict(rows=rows, metadata=metadata), ensure_ascii=False).replace("<", "\\u003c")
    return '''<!doctype html><html lang="es"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Jade · Análisis de partida</title><style>
*{box-sizing:border-box}body{margin:0;background:#f5f6ef;color:#173c30;font:15px/1.5 system-ui,sans-serif}
main{max-width:1050px;margin:auto;padding:16px}h1{margin:0 0 8px}p{margin:8px 0}.layout{display:grid;grid-template-columns:minmax(220px,400px) minmax(0,1fr);gap:20px}.layout>section{min-width:0}
.board-eval{display:grid;grid-template-columns:30px minmax(0,1fr);gap:8px;align-items:stretch}.evalbar{position:relative;overflow:hidden;border:1px solid #788479;border-radius:6px;background:#23272c;min-height:120px}.evalwhite{position:absolute;bottom:0;width:100%;height:50%;background:#fffdf4;transition:height .35s ease}.evalzero{position:absolute;top:50%;width:100%;border-top:1px dashed #8e998d}.eval-label{position:absolute;left:0;right:0;bottom:6px;text-align:center;writing-mode:vertical-rl;transform:rotate(180deg);font-weight:700;font-size:12px;color:#101820;background:#fffdf4;padding:3px;border-radius:3px}.eval-caption{font-size:13px;min-height:40px}#board{min-width:0}#board svg{width:100%;height:auto;display:block}button,select{font:inherit;border:1px solid #bbccbf;border-radius:8px;background:white;color:#173c30;padding:10px;cursor:pointer;min-height:44px}
.controls{display:flex;gap:6px;flex-wrap:wrap;margin:12px 0}.scroll{overflow:auto;max-height:540px;background:white;border-radius:10px}table{width:100%;border-collapse:collapse;font-size:13px;white-space:nowrap}
th,td{text-align:left;padding:9px;border-bottom:1px solid #e2e9db}tr[data-i]{cursor:pointer}tr.selected{background:#dceac8}tr:hover{background:#edf2e3}.note{color:#4f6759;font-size:13px}
#explain{padding:12px;border-radius:10px;background:#e5eddf;min-height:110px;overflow-wrap:anywhere}.bad{color:#a0392f}@media(max-width:690px){.layout{grid-template-columns:minmax(0,1fr)}main{padding:12px}.scroll{max-height:340px}}
</style><main><h1>Jade · Análisis con Stockfish</h1><p id="game"></p><p class="note" id="meta"></p><p class="bad" id="warning"></p>
<p class="note">Evaluaciones desde el punto de vista de las blancas: positivo favorece a blancas, negativo a negras. Las etiquetas son heurísticas propias. WDL estima resultados del motor, no tus probabilidades reales de ganar.</p>
<div class="layout"><section><div class="board-eval"><div id="evalbar" class="evalbar" role="img" aria-label="Evaluación"><div id="evalwhite" class="evalwhite"></div><div class="evalzero"></div><span id="eval-label" class="eval-label">?</span></div><div id="board"></div></div><p id="eval-caption" class="eval-caption" aria-live="polite"></p><div class="controls"><button id="prev">Anterior</button><button id="next">Siguiente</button><button id="line">Ver jugada de Stockfish</button><button id="back">Antes / después</button></div><div id="explain"></div></section>
<section><div class="controls"><label>Ver <select id="side"><option value="">Ambos colores</option><option value="Blancas">Blancas</option><option value="Negras">Negras</option></select></label></div><div class="scroll"><table><thead><tr><th>Jugada</th><th>Valoración</th><th>Mejor</th><th>Eval.</th><th>Pérdida cp</th></tr></thead><tbody id="rows"></tbody></table></div></section></div></main>
<script>(()=>{const d=__DATA__,e=id=>document.getElementById(id);let index=0,mode='played';
e('game').textContent=d.metadata.white+' contra '+d.metadata.black+' · '+d.metadata.result;
e('meta').textContent=d.metadata.engine+' · '+d.metadata.seconds+' s por búsqueda · '+d.metadata.threads+' hilos · '+d.rows.length+'/'+d.metadata.total+' jugadas analizadas.';
e('warning').textContent=d.metadata.complete?'':'ANÁLISIS INCOMPLETO. '+(d.metadata.error||'Se interrumpió la ejecución.');
function render(){const r=d.rows[index];const score=mode==='before'?r.best_score:mode==='best'?(r.best_after_score||r.best_score):r.played_score;let cp=null,mate=null,value=0,text=mode==='before'?(r.best_evaluation||'?'):mode==='best'?(r.best_after_evaluation||r.best_evaluation||'?'):r.evaluation;
if(score){cp=score.cp;mate=score.mate;value=score.value}else if(text&&!text.includes('#')){cp=Number(text)*100}
let share=.5,detail='Evaluación no disponible';if(mate!==null){share=value>0?1:0;detail='Mate '+(value>0?'a favor de blancas':'a favor de negras')+' · '+Math.abs(mate)}else if(cp!==null&&Number.isFinite(cp)){share=.5+.5*Math.tanh(cp/400);detail=(cp>0?'+':'')+cp+' cp ('+(cp/100).toFixed(2)+' peones)'}
e('evalwhite').style.height=(share*100)+'%';e('eval-label').textContent=text;e('evalbar').setAttribute('aria-label',detail);e('eval-caption').textContent=(mode==='played'?'Tras tu jugada: ':mode==='before'?'Antes de mover: ':'Tras la alternativa de Stockfish: ')+detail+'. Barra orientativa, no probabilidad de victoria.';
e('board').innerHTML=mode==='played'?(r.after_svg||r.svg):mode==='before'?r.svg:(r.best_after_svg||r.pv_svg);}
function show(i){if(!d.rows.length){e('explain').textContent='No hay jugadas analizadas.';return}index=Math.max(0,Math.min(d.rows.length-1,i));mode='played';const r=d.rows[index];render();
e('explain').replaceChildren();const a=document.createElement('b');a.textContent=r.turn+' '+r.played+' · '+r.label;const b=document.createElement('p');b.textContent='Stockfish prefiere '+r.best+'. Continuación: '+r.pv;const c=document.createElement('p');c.textContent='Evaluación de tu jugada: '+r.evaluation+'; profundidad '+r.depth+'.'+(r.loss_cp===null?' Mate: no se convierte a centipeones ficticios.':' Pérdida estimada: '+r.loss_cp+' cp.');e('explain').append(a,b,c);
e('rows').querySelectorAll('tr').forEach(tr=>tr.classList.toggle('selected',Number(tr.dataset.i)===index));}
function table(){e('rows').replaceChildren();d.rows.forEach((r,i)=>{if(e('side').value&&r.color!==e('side').value)return;const tr=document.createElement('tr');tr.dataset.i=i;[r.turn+' '+r.played,r.label,r.best,r.evaluation,r.loss_cp===null?'mate':r.loss_cp].forEach(t=>{const td=document.createElement('td');td.textContent=t;tr.append(td)});tr.onclick=()=>show(i);e('rows').append(tr)});show(index)}
e('side').onchange=table;e('prev').onclick=()=>show(index-1);e('next').onclick=()=>show(index+1);e('back').onclick=()=>{if(!d.rows.length)return;mode=mode==='before'?'played':'before';render()};e('line').onclick=()=>{if(!d.rows.length)return;mode=mode==='best'?'played':'best';render()};table();})();</script></html>'''.replace("__DATA__", payload)


def analizar_pgn_jade(text, binary, output_dir, seconds=3., threads=None, hash_mb=512, log=print):
    """Analiza una partida con un proceso independiente a máxima fuerza configurada.

    Cada jugada compara búsqueda libre y búsqueda forzando la jugada real, con
    el mismo tiempo por búsqueda. No usa la temperatura, el reloj ni los datos
    humanos de Jade. El tiempo es finito: ninguna profundidad garantiza perfección.
    """
    seconds = float(seconds)
    if not .01 <= seconds <= 120:
        raise ValueError("Elige entre 0,01 y 120 segundos por búsqueda.")
    stream = io.StringIO(str(text).lstrip("\ufeff"))
    game = chess.pgn.read_game(stream)
    if game is None or game.errors:
        raise ValueError("No se pudo leer el PGN completo. Revisa las jugadas y la posición inicial.")
    if chess.pgn.read_game(stream) is not None:
        raise ValueError("El archivo contiene varias partidas. Exporta una sola para este analizador.")
    board = game.board()
    if not board.is_valid() or board.chess960 or game.headers.get("Variant", "Standard") not in ("Standard", "Chess"):
        raise ValueError("Este analizador admite ajedrez estándar con posición inicial válida.")
    moves = list(game.mainline_moves())
    if not moves:
        raise ValueError("El PGN no contiene movimientos.")
    thread_count = max(1, min(int(threads or min(os.cpu_count() or 1, 4)), os.cpu_count() or 1))
    hash_mb = max(16, min(int(hash_mb), 2048))
    folder = Path(output_dir) / ("analisis_" + time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6])
    folder.mkdir(parents=True, exist_ok=False)
    annotated = chess.pgn.Game()
    annotated.headers.update(game.headers)
    annotated.setup(game.board())
    annotated.headers["Annotator"] = "Jade · Stockfish sin humanización"
    node, rows = annotated, []
    metadata = dict(white=game.headers.get("White", "Blancas"), black=game.headers.get("Black", "Negras"),
                    result=game.headers.get("Result", "*"), total=len(moves), seconds=seconds,
                    threads=thread_count, complete=False, error="", engine="Stockfish")

    def export():
        metadata["complete"] = len(rows) == len(moves)
        annotated.headers["Result"] = metadata["result"] if metadata["complete"] else "*"
        (folder / "partida_analizada.pgn").write_text(str(annotated), encoding="utf-8")
        fields = ["turn", "color", "played", "best", "evaluation", "loss_cp", "loss_wdl", "label", "depth", "pv"]
        with (folder / "movimientos.csv").open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        (folder / "informe.html").write_text(jade_report_html(rows, metadata), encoding="utf-8")

    engine = None
    try:
        engine = chess.engine.SimpleEngine.popen_uci(str(binary), timeout=seconds + 60)
        engine.configure({"Threads": thread_count, "Hash": hash_mb, "Skill Level": 20,
                          "UCI_LimitStrength": False})
        metadata["engine"] = engine.id.get("name", "Stockfish")
        token = object()
        log(f"{metadata['engine']} a máxima fuerza configurada. Hasta ~{len(moves) * 2 * seconds / 60:.1f} min de búsqueda, más exportación.")
        original_nodes = list(game.mainline())
        for index, move in enumerate(moves):
            mover = board.turn
            limit = chess.engine.Limit(time=seconds)
            best_info = engine.analyse(board, limit, multipv=1, game=token)[0]
            best_move = best_info["pv"][0]
            played_info = best_info if move == best_move else engine.analyse(board, limit, root_moves=[move], game=token)
            best_score = best_info["score"].pov(mover)
            played_score = played_info["score"].pov(mover)
            label, cp_loss, wdl_loss = jade_grade(best_score, played_score, move == best_move, board.ply())
            if board.legal_moves.count() == 1:
                label = "Única jugada legal"
            pv = best_info["pv"][:8]
            continuation = board.variation_san(pv)
            future = board.copy(stack=True)
            for next_move in pv:
                future.push(next_move)
            san, best_san = board.san(move), board.san(best_move)
            turn = f"{board.fullmove_number}." if mover else f"{board.fullmove_number}..."
            arrows = [chess.svg.Arrow(best_move.from_square, best_move.to_square, color="#21885ac0")]
            if move != best_move:
                arrows.append(chess.svg.Arrow(move.from_square, move.to_square, color="#c8503ab0"))
            row = dict(turn=turn, color="Blancas" if mover else "Negras", played=san, best=best_san,
                       evaluation=jade_score_text(played_info["score"].white()),
                       loss_cp=cp_loss, loss_wdl=round(wdl_loss, 4), label=label,
                       depth=played_info.get("depth", 0), pv=continuation,
                       svg=chess.svg.board(board, arrows=arrows), pv_svg=chess.svg.board(future, lastmove=pv[-1]))
            def numeric_score(info):
                score=info["score"].white()
                return dict(cp=score.score(),mate=score.mate(),value=score.score(mate_score=100000))
            after=board.copy(stack=True);after.push(move)
            best_after=board.copy(stack=True);best_after.push(best_move)
            row.update(played_score=numeric_score(played_info),best_score=numeric_score(best_info),
                       best_evaluation=jade_score_text(best_info["score"].white()),
                       after_svg=chess.svg.board(after,lastmove=move),
                       best_after_svg=chess.svg.board(best_after,lastmove=best_move))
            # La posición terminal ya no tiene un «mate en 1» pendiente.
            def resulting_score(position,fallback):
                if position.is_checkmate():
                    return {"score":chess.engine.PovScore(chess.engine.Mate(0),position.turn)}
                if position.is_game_over():
                    return {"score":chess.engine.PovScore(chess.engine.Cp(0),chess.WHITE)}
                return fallback
            actual_result=resulting_score(after,played_info)
            best_result=resulting_score(best_after,best_info)
            row.update(played_score=numeric_score(actual_result),evaluation=jade_score_text(actual_result["score"].white()),
                       best_after_score=numeric_score(best_result),best_after_evaluation=jade_score_text(best_result["score"].white()))
            rows.append(row)
            node = node.add_variation(move)
            node.comment = f"Jade: {label}. Mejor: {best_san}. Línea: {continuation}."
            node.set_eval(played_info["score"], played_info.get("depth"))
            original_clock = original_nodes[index].clock()
            if original_clock is not None:
                node.set_clock(original_clock)
            if move != best_move:
                alternative = node.parent.add_line(pv)
                alternative.comment = "Continuación preferida por Stockfish."
            board.push(move)
            log(f"{index+1}/{len(moves)} · {turn} {san}: {label}; mejor {best_san}")
            if (index + 1) % 5 == 0:
                export()
    except KeyboardInterrupt:
        metadata["error"] = "Interrumpido por el usuario; se conserva el análisis completado."
        log(metadata["error"])
    except Exception as exc:
        metadata["error"] = str(exc)
        log("Análisis incompleto: " + str(exc))
    finally:
        if engine is not None:
            try:
                engine.quit()
            except chess.engine.EngineTerminatedError:
                pass
        export()
    return dict(rows=rows, metadata=metadata, directory=folder,
                html=folder / "informe.html", pgn=folder / "partida_analizada.pgn", csv=folder / "movimientos.csv")


def abrir_analizador_jade(jade, tablero=None, output_dir=None):
    """Formulario para pegar PGN, subir un archivo o cargar la partida de Jade."""
    import ipywidgets as widgets
    if output_dir is None:
        game_path = getattr(tablero, "game_path", None)
        output_dir = (game_path.parent.parent if game_path else Path("/content/Jade")) / "analisis"
    text = widgets.Textarea(placeholder="Pega aquí el PGN completo de Lichess o Chess.com",
                           layout=widgets.Layout(width="100%", height="160px"))
    upload = widgets.FileUpload(accept=".pgn,.txt", multiple=False, description="Subir PGN")
    current = widgets.Button(description="Usar partida de Jade", layout=widgets.Layout(width="100%"))
    seconds = widgets.Dropdown(options=[("Rápido: 1 s", 1), ("Normal: 3 s", 3), ("Profundo: 10 s", 10), ("Muy profundo: 30 s", 30)], value=3,
                               description="Análisis:", layout=widgets.Layout(width="100%"))
    run = widgets.Button(description="Analizar con Stockfish", button_style="success", layout=widgets.Layout(width="100%"))
    out = widgets.Output(layout=widgets.Layout(width="100%"))

    def use_current(_):
        if tablero is not None:
            if hasattr(tablero, "settle_clock"):
                tablero.settle_clock()
                tablero.paused, tablero.clock_started = True, None
                tablero.save_game()
            text.value = tablero.pgn()

    def uploaded(change):
        items = change["new"]
        if items:
            first = next(iter(items.values())) if isinstance(items, dict) else items[0]
            try:
                text.value = bytes(first["content"]).decode("utf-8-sig")
            except UnicodeDecodeError:
                text.value = bytes(first["content"]).decode("latin-1")

    def analyse(_):
        run.disabled = True
        with out:
            out.clear_output()
            try:
                if tablero is not None and hasattr(tablero, "settle_clock"):
                    tablero.settle_clock()
                    tablero.paused, tablero.clock_started = True, None
                    tablero.save_game()
                result = analizar_pgn_jade(text.value, jade.binary, output_dir, seconds=seconds.value)
                print("Archivos guardados en:", result["directory"])
                mostrar_informe_jade(result["html"])
                # Descargas gestionadas por Colab, disponibles también para móvil.
                for name in ("html", "pgn", "csv"):
                    button = widgets.Button(description="Descargar " + name.upper())
                    def download(_, path=result[name]):
                        from google.colab import files
                        files.download(str(path))
                    button.on_click(download)
                    display(button)
            except Exception as exc:
                print("No se pudo analizar:", exc)
            finally:
                run.disabled = False

    current.on_click(use_current)
    upload.observe(uploaded, names="value")
    run.on_click(analyse)
    root = widgets.VBox([widgets.HTML("<b>Analizar PGN con Stockfish</b><br>Pega el texto, sube un archivo o usa tu partida. Se pausará el reloj de práctica. Las etiquetas son orientativas."),
                         current, upload, text, seconds, run, out],
                        layout=widgets.Layout(width="100%", max_width="800px", min_width="0"))
    display(root)
    return root


def mostrar_informe_jade(path):
    """Iframe propio: el JavaScript no depende del renderizador de Output."""
    path=Path(path)
    content=html.escape(path.read_text(encoding="utf-8"),quote=True)
    display(HTML('<iframe title="Análisis de Jade" sandbox="allow-scripts" '
                 f'srcdoc="{content}" style="width:100%;height:1200px;border:0"></iframe>'))


def abrir_ultimo_informe_jade(folder):
    files=list(Path(folder).glob("analisis_*/informe.html"))
    if not files:
        raise FileNotFoundError("No hay informes guardados en "+str(folder))
    path=max(files,key=lambda p:p.stat().st_mtime)
    print("Informe:",path.parent.name)
    mostrar_informe_jade(path)

def memory_report(memory):
    report=memory.summary()
    print(f"Jade 2.2.1 · {memory.rhythm}: {report['partidas']:,} partidas registradas; {report['base_MB']:.2f} MB locales")
    print('Corpus nuevo:',report['corpus'],'| Antiguas conservadas:',report['legacy_games'])
    print('Perfil | Partidas que alimentan frecuencias | Movimientos')
    for p in PROFILES:
        s=report['perfiles'].get(p,{'partidas':0,'movimientos':0})
        print(p,s['partidas'],s['movimientos'])
    return report

JADE_LIGHT_SOURCE='"""Jade 2.2.1 en CPU: requiere numpy, python-chess y un Stockfish local."""\nimport argparse, hashlib, json, math, sqlite3, time\nfrom pathlib import Path\nimport numpy as np\nimport chess, chess.engine\nPROFILES=(1200,1300,1400,1500,1600)\nJADE_FEATURES=1560\ndef rating_band(rating):\n    return 100*math.floor((int(rating)+50)/100)\ndef pack_position(board):\n    """Clave exacta equivalente a FEN4: piezas, turno, enroque y EP legal."""\n    pieces = bytearray(32)\n    for square, piece in board.piece_map().items():\n        value = piece.piece_type + (0 if piece.color else 6)\n        pieces[square // 2] |= value << (4 * (square % 2))\n    flags = int(board.turn)\n    for i, (color, kingside) in enumerate(((True, True), (True, False), (False, True), (False, False)), 1):\n        allowed = board.has_kingside_castling_rights(color) if kingside else board.has_queenside_castling_rights(color)\n        flags |= int(allowed) << i\n    ep = board.ep_square + 1 if board.has_legal_en_passant() else 0\n    return bytes(pieces) + bytes((flags, ep))\n\ndef unpack_move(value):\n    return chess.Move(value & 63, (value >> 6) & 63, promotion=(value >> 12) or None)\n\ndef jade_position_id(conn,key,create=False):\n    """ID de 63 bits con comparación EXACTA y resolución explícita de colisiones.\n\n    Evita un segundo índice que repetiría los 34 bytes de cada posición. El hash\n    nunca se usa como prueba de igualdad: se comprueba siempre la clave completa.\n    """\n    for salt in range(1000):\n        pid=int.from_bytes(hashlib.blake2b(key+salt.to_bytes(4,"little"),digest_size=8).digest(),"little") & ((1<<63)-1)\n        row=conn.execute("SELECT key FROM positions WHERE id=?",(pid,)).fetchone()\n        if row is None:\n            if not create:\n                return None\n            conn.execute("INSERT OR IGNORE INTO positions(id,key) VALUES (?,?)",(pid,key))\n            row=conn.execute("SELECT key FROM positions WHERE id=?",(pid,)).fetchone()\n        if row[0]==key:\n            return pid\n    raise RuntimeError("No se pudo asignar un identificador de posición sin colisión.")\ndef jade_context(board, profile, opponent=None, remaining=None, other_remaining=None,\n                 initial=None, increment=None, rhythm="rapid", history_known=True):\n    return dict(profile=profile,opponent=opponent,remaining=remaining,other_remaining=other_remaining,\n                initial=initial,increment=increment,rhythm=rhythm,history_known=history_known)\n\ndef encode_jade(board, context):\n    """Solo información anterior al movimiento objetivo; perspectiva del que mueve."""\n    features=np.zeros(JADE_FEATURES,dtype=np.float32)\n    side=board.turn\n    previous=board.copy(stack=1)\n    if previous.move_stack:\n        previous.pop()\n    for offset,position in ((0,board),(768,previous)):\n        for color,base in ((side,0),(not side,6)):\n            for piece_type in chess.PIECE_TYPES:\n                squares=position.pieces_mask(piece_type,color)\n                for square in chess.scan_forward(squares):\n                    oriented=square if side else square^56\n                    features[offset+(base+piece_type-1)*64+oriented]=1\n    c=features[1536:]\n    def known(value):\n        return value is not None and math.isfinite(float(value)) and float(value)>=0\n    c[0]=min(2,max(0,float(context["profile"])/2000))\n    if known(context.get("opponent")):\n        c[1],c[2]=min(2,max(0,float(context["opponent"])/2000)),1\n    initial=context.get("initial")\n    denominator=float(initial) if known(initial) and initial>0 else 600.\n    for value_index,flag_index,name in ((3,4,"remaining"),(5,6,"other_remaining")):\n        value=context.get(name)\n        if known(value):\n            c[value_index],c[flag_index]=min(4,max(0,float(value)/denominator)),1\n    for value_index,flag_index,name,scale in ((7,8,"initial",7200),(9,10,"increment",60)):\n        value=context.get(name)\n        if known(value):\n            c[value_index],c[flag_index]=min(2,math.log1p(value)/math.log1p(scale)),1\n    c[11:14]=[min(board.ply()/100,3),min(board.halfmove_clock/100,2),float(board.is_repetition(2))]\n    c[14:18]=[board.has_kingside_castling_rights(side),board.has_queenside_castling_rights(side),\n               board.has_kingside_castling_rights(not side),board.has_queenside_castling_rights(not side)]\n    c[18]=(chess.square_file(board.ep_square)+1)/8 if board.has_legal_en_passant() else 0\n    c[19:24]=[context.get("rhythm")=="blitz",bool(board.move_stack),context.get("history_known",True),board.is_check(),bool(board.move_stack)]\n    legal=list(board.legal_moves)\n    codes=np.array([[m.from_square if side else m.from_square^56,\n                     m.to_square if side else m.to_square^56,m.promotion or 0] for m in legal],dtype=np.int64).reshape(-1,3)\n    return features,codes,legal\n\nclass NumpyJadePolicy:\n    """No importa PyTorch ni necesita GPU. Carga solo pesos y metadatos JSON."""\n    def __init__(self,weights,metadata=None):\n        self.weights=weights\n        self.metadata=metadata or {}\n        self.alpha=float(self.metadata.get("alpha",0))\n        self.profiles=set(self.metadata.get("profiles",[]))\n        self.active=bool(self.metadata.get("approved",False))\n    @classmethod\n    def load(cls,path,metadata=None):\n        with np.load(str(path),allow_pickle=False) as archive:\n            weights={key:archive[key] for key in archive.files}\n        return cls(weights,metadata)\n    def linear(self,x,name):\n        w=self.weights\n        if name+".q" in w:\n            value=(x@w[name+".q"].T)*w[name+".scale"]\n        else:\n            value=x@w[name+".weight"].T\n        return value+w[name+".bias"]\n    def probabilities(self,board,context):\n        features,moves,legal=encode_jade(board,context)\n        if not legal:\n            return {}\n        probabilities=self.probabilities_encoded(features,moves)\n        return dict(zip((m.uci() for m in legal),probabilities.tolist()))\n    def probabilities_encoded(self,features,moves):\n        hidden=np.maximum(0,self.linear(np.maximum(0,self.linear(features,"fc1")),"fc2"))\n        w=self.weights\n        combined=np.concatenate((np.broadcast_to(hidden,(len(moves),len(hidden))),w["source.weight"][moves[:,0]],\n                                  w["destination.weight"][moves[:,1]],w["promotion.weight"][moves[:,2]]),axis=1)\n        logits=self.linear(np.maximum(0,self.linear(combined,"move1")),"move2").ravel().astype(np.float64)\n        probabilities=np.exp(logits-logits.max());probabilities/=probabilities.sum()\n        return probabilities\n    def probabilities_batch(self,examples):\n        """Evaluación por bloques acotados; reutiliza características ya calculadas."""\n        lengths=np.array([len(x["codes"]) for x in examples])\n        features=np.stack([x["features"] for x in examples])\n        moves=np.concatenate([x["codes"] for x in examples])\n        hidden=np.maximum(0,self.linear(np.maximum(0,self.linear(features,"fc1")),"fc2"))\n        w=self.weights\n        combined=np.concatenate((np.repeat(hidden,lengths,axis=0),w["source.weight"][moves[:,0]],\n                                 w["destination.weight"][moves[:,1]],w["promotion.weight"][moves[:,2]]),axis=1)\n        logits=self.linear(np.maximum(0,self.linear(combined,"move1")),"move2").ravel().astype(np.float64)\n        result=[];offset=0\n        for length in lengths:\n            values=logits[offset:offset+length];p=np.exp(values-values.max());p/=p.sum()\n            result.append(p);offset+=length\n        return result\n"""Jade 1.2: candidatas humanas y de motor, presupuesto de tiempo y muestreo."""\nimport math\nimport time\nimport numpy as np\nimport chess\nimport chess.engine\n\n\nclass JadeEngine:\n    VERSION = "1.2"\n\n    def __init__(self, binary, memory, nodes=100_000):\n        self.binary, self.memory, self.nodes = str(binary), memory, int(nodes)\n        self.engine = chess.engine.SimpleEngine.popen_uci(self.binary, timeout=60)\n        self.engine.configure({"Threads": 1, "Hash": 128, "Skill Level": 20,\n                               "UCI_LimitStrength": False})\n        self.rng = np.random.default_rng()\n        self.game_token = object()\n\n    def new_game(self):\n        self.game_token = object()\n        self.engine.configure({"Clear Hash": None})\n\n    @staticmethod\n    def candidate_limit(seconds=None, maximum=15):\n        if maximum not in (12, 15):\n            raise ValueError("El máximo de candidatas debe ser 12 o 15.")\n        if seconds is None or seconds > 180:\n            return maximum\n        if seconds > 90:\n            return min(maximum, 12)\n        if seconds > 30:\n            return 8\n        if seconds > 10:\n            return 5\n        return 3\n\n    @staticmethod\n    def search_seconds(remaining=None):\n        # Presupuesto TOTAL de búsqueda, incluidas las comprobaciones humanas.\n        if remaining is None or remaining > 180:\n            return .9\n        if remaining > 90:\n            return .6\n        if remaining > 30:\n            return .35\n        if remaining > 10:\n            return .18\n        return max(.005, min(.08, remaining * .10))\n\n    def human_counts(self, board, profile):\n        legal = {m.uci() for m in board.legal_moves}\n        return {uci: float(n) for uci, n in self.memory.counts_for(board, profile).items()\n                if uci in legal and n > 0}\n\n    @staticmethod\n    def _row(board, info, counts):\n        move = info["pv"][0]\n        score = info["score"].pov(board.turn)\n        return dict(move=move, uci=move.uci(), san=board.san(move),\n                    cp=score.score(), mate=score.mate(), value=score.score(mate_score=100000),\n                    human_count=counts.get(move.uci(), 0), depth=info.get("depth", 0))\n\n    def candidates(self, board, profile=1300, remaining=None, maximum=15, use_humans=True):\n        if not board.is_valid():\n            raise ValueError("Posición no válida.")\n        if board.is_game_over():\n            return []\n        k = min(self.candidate_limit(remaining, maximum), board.legal_moves.count())\n        counts = self.human_counts(board, profile) if use_humans else {}\n        # Se reservan plazas a jugadas reales, incluso fuera del ranking del motor.\n        ranked = sorted(counts, key=lambda uci: (-counts[uci], uci))\n        opening = board.ply() < 24\n        known = [chess.Move.from_uci(uci) for uci in ranked if counts[uci] >= 2]\n        human_slots = min(k - 1, (k * 2 // 3) if opening else (k // 2))\n        humans = known[:human_slots]\n        budget = self.search_seconds(remaining)\n        begun = time.monotonic()\n        if not humans:\n            infos = self.engine.analyse(board, chess.engine.Limit(time=budget, nodes=self.nodes),\n                                        multipv=k, game=self.game_token)\n            rows = [self._row(board, info, counts) for info in infos if info.get("pv") and "score" in info]\n            for row in rows:\n                row["outside_engine_list"] = False\n            return sorted(rows, key=lambda row: row["value"], reverse=True)\n\n        # Primera búsqueda libre: conserva una referencia táctica y completa plazas.\n        preliminary = self.engine.analyse(board, chess.engine.Limit(time=budget * .35, nodes=max(100, self.nodes // 3)),\n                                          multipv=k, game=self.game_token)\n        engine_moves = [info["pv"][0] for info in preliminary if info.get("pv")]\n        if not engine_moves:\n            raise RuntimeError("Stockfish no devolvió candidatas.")\n        pool = list(dict.fromkeys([engine_moves[0]] + humans + engine_moves))[:k]\n        # Un segundo análisis compara TODAS las candidatas, incluidas las humanas.\n        left = max(.005, budget - (time.monotonic() - begun))\n        infos = self.engine.analyse(board, chess.engine.Limit(time=left, nodes=max(100, self.nodes * 2 // 3)),\n                                    multipv=len(pool), root_moves=pool, game=self.game_token)\n        rows = [self._row(board, info, counts) for info in infos if info.get("pv") and "score" in info]\n        for row in rows:\n            row["outside_engine_list"] = row["move"] not in engine_moves\n        return sorted(rows, key=lambda row: row["value"], reverse=True)\n\n    def distribution(self, board, candidates, profile=1300, beta=1.0, temperature=None, remaining=None):\n        if profile not in (1200, 1300, 1400, 1500, 1600):\n            raise ValueError("Perfil no disponible.")\n        t = float(np.interp(profile, [1200, 1600], [180., 45.])) if temperature is None else float(temperature)\n        if not math.isfinite(t) or t <= 0 or not math.isfinite(beta) or not 0 <= beta <= 2:\n            raise ValueError("Temperatura positiva y peso humano entre 0 y 2.")\n        if not candidates:\n            raise ValueError("No hay candidatas.")\n        # Con poco tiempo se reduce el cálculo y aumenta la dispersión, aunque\n        # haya menos candidatas. No se obliga a escoger siempre la primera.\n        pressure = 1. if remaining is None or remaining > 90 else 1.2 if remaining > 30 else 1.5 if remaining > 10 else 1.9\n        t *= pressure\n        counts = self.human_counts(board, profile)\n        values = np.array([row["value"] for row in candidates], dtype=float)\n        human = np.array([counts.get(row["uci"], 0) for row in candidates], dtype=float)\n        loss = np.maximum(0., values.max() - values)\n        weights = np.exp(np.clip((values - values.max()) / t, -700, 0))\n        p_engine = weights / weights.sum()\n        opening = board.ply() < 24\n        # Tolerancia heurística: admite gambitos y elecciones subóptimas comunes.\n        free_cp = float(np.interp(profile, [1200, 1600], [220., 100.])) * (1.25 if opening else 1.)\n        penalty = np.exp(-np.maximum(0., loss - free_cp) / (120. * pressure))\n        best_mate = candidates[int(np.argmax(values))]["mate"]\n        for i, row in enumerate(candidates):\n            if row["mate"] is not None and row["mate"] < 0 and (best_mate is None or best_mate >= 0):\n                penalty[i] = 0.  # Evita un mate detectado si hay una alternativa.\n            elif best_mate is not None and best_mate > 0 and row["mate"] is None:\n                # No convierte un mate positivo en una diferencia ficticia de cp.\n                penalty[i] = math.exp(-max(0., 350. - (row["cp"] or 0)) / 200.)\n        evidence = float(human.sum())\n        p_human = human * penalty\n        mix = ((beta / (beta + .25)) * (evidence / (evidence + 15.)) * (1. if opening else .7)) if beta else 0.\n        if p_human.sum() <= 0:\n            mix, p_human = 0., p_engine.copy()\n        else:\n            p_human /= p_human.sum()\n        probabilities = (1. - mix) * p_engine + mix * p_human\n        details = dict(temperature=t, human_mix=float(mix), candidate_count=len(candidates),\n                       candidate_observations=int(evidence), position_observations=sum(counts.values()),\n                       # Alias para abrir partidas guardadas por la interfaz anterior.\n                       top5_observations=int(evidence), opening=opening)\n        return probabilities, details\n\n    def reaction_seconds(self, board, candidates, details, remaining=None, pace=1.):\n        if pace < 0 or not math.isfinite(pace):\n            raise ValueError("El ritmo de respuesta debe ser positivo.")\n        if board.legal_moves.count() == 1:\n            target = self.rng.uniform(.4, .9)\n        elif remaining is not None and remaining <= 10:\n            target = self.rng.uniform(.15, .55)\n        elif remaining is not None and remaining <= 30:\n            target = self.rng.uniform(.4, 1.3)\n        elif remaining is not None and remaining <= 90:\n            target = self.rng.uniform(1., 3.)\n        else:\n            target = self.rng.uniform(2.5, 6.)\n            close = sum(candidates[0]["value"] - row["value"] < 70 for row in candidates)\n            target *= 1. + min(close, 5) * .10\n            if details["opening"] and details["position_observations"] >= 30:\n                target *= .75\n        target = min(10., target * pace)\n        if remaining is not None:\n            target = min(target, max(0., remaining * .18))\n        return float(target)\n\n    def choose(self, board, profile=1300, beta=1., temperature=None, remaining=None, maximum=15, pace=1.):\n        candidates = self.candidates(board, profile, remaining, maximum, use_humans=beta > 0)\n        p, details = self.distribution(board, candidates, profile, beta, temperature, remaining)\n        index = int(self.rng.choice(len(candidates), p=p))\n        row = candidates[index]\n        details.update(selected_san=row["san"], selected_human_count=row["human_count"],\n                       outside_engine_list=row["outside_engine_list"], probability=float(p[index]),\n                       max_candidates=self.candidate_limit(remaining, maximum))\n        details["reaction_seconds"] = self.reaction_seconds(board, candidates, details, remaining, pace)\n        return row["move"], details\n\n    def close(self):\n        try:\n            self.engine.quit()\n        except chess.engine.EngineTerminatedError:\n            pass\n\nClassicJadeEngine=JadeEngine\n"""Integra el predictor aprobado sin reemplazar Stockfish ni la memoria humana."""\n\n\nclass JadeEngine(ClassicJadeEngine):\n    VERSION="2.2.1"\n\n    def __init__(self,binary,memory,nodes=100000,policy=None,hash_mb=64):\n        super().__init__(binary,memory,nodes)\n        self.policy=policy\n        self.engine.configure({"Hash":max(16,min(512,int(hash_mb)))})\n        self.prediction={}\n        self.prediction_key=None\n\n    def choose(self,board,profile=1300,beta=1.,temperature=None,remaining=None,maximum=15,pace=1.):\n        self.prediction={}\n        self.prediction_key=(board.fen(),profile)\n        if self.policy and self.policy.active and profile in self.policy.profiles and beta>0:\n            context=jade_context(board,profile,remaining=remaining,rhythm=self.memory.rhythm)\n            self.prediction=self.policy.probabilities(board,context)\n        # Una búsqueda de una sola línea da una referencia más profunda que\n        # repartir todo el cálculo entre 15 candidatas. Cuenta en el reloj real.\n        self.tactical_anchor=None\n        self.anchor_fen=board.fen()\n        if not board.is_game_over():\n            budget=self.search_seconds(remaining)*.4\n            info=self.engine.analyse(board,chess.engine.Limit(time=max(.005,budget),\n                                     nodes=max(100,self.nodes//2)),game=self.game_token)\n            if info.get("pv") and "score" in info:\n                self.tactical_anchor=self._row(board,info,{})\n        move,details=super().choose(board,profile,beta,temperature,remaining,maximum,pace)\n        anchor=self.tactical_anchor\n        # Confirma la decisión contra la referencia con solo DOS variantes.\n        # Se usa el mismo control tras TODAS las mezclas de memoria y red.\n        if anchor and move!=anchor["move"]:\n            budget=max(.005,self.search_seconds(remaining)*.4)\n            infos=self.engine.analyse(board,chess.engine.Limit(time=budget,nodes=max(100,self.nodes//2)),\n                multipv=2,root_moves=[anchor["move"],move],game=self.game_token)\n            checked=[self._row(board,x,{}) for x in infos if x.get("pv") and "score" in x]\n            by_move={x["move"]:x for x in checked}\n            if len(by_move)==2:\n                rows=list(by_move.values());weights=np.array([float(x["move"]==move) for x in rows])\n                safe,guard=self.guard_distribution(rows,weights,profile)\n                if safe[rows.index(by_move[move])]==0:\n                    move=rows[int(np.argmax(safe))]["move"]\n                    details.update(selected_san=board.san(move),probability=1.,\n                                   safety_override=True,selected_human_count=self.human_counts(board,profile).get(move.uci(),0),\n                                   outside_engine_list=False)\n                details["verification_loss_cp"]=max(x["value"] for x in rows)-by_move[move]["value"]\n                details["verified"]=True\n        return move,details\n\n    @staticmethod\n    def guard_distribution(candidates,probabilities,profile):\n        """Límite de pérdidas estimadas. Heurístico: NO certifica un ELO.\n\n        No usa un porcentaje de error por movimiento que fuerce fallos sucesivos.\n        Imprecisiones plausibles siguen permitidas; nunca fuerza un error.\n        """\n        values=np.array([r["value"] for r in candidates],dtype=float)\n        loss=np.maximum(0.,values.max()-values)\n        cap=float(np.interp(profile,[1200,1600],[140.,65.]))\n        best=candidates[int(np.argmax(values))]\n        mate_wins=lambda r:r.get("mate") is not None and r["value"]>0\n        mate_loses=lambda r:r.get("mate") is not None and r["value"]<0\n        allowed=loss<=cap\n        for i,row in enumerate(candidates):\n            if mate_wins(best):\n                # Evita aplazar una y otra vez el mate por el muestreo humano.\n                allowed[i]=mate_wins(row) and row["mate"]<=best["mate"]\n            elif mate_loses(best):\n                allowed[i]=True  # Si todas pierden, no inventa una salida.\n            elif mate_loses(row):\n                allowed[i]=False\n        p=np.asarray(probabilities,dtype=float).copy()\n        p[~allowed]=0.\n        # Dentro del margen, desincentiva pérdidas continuas de medio peón.\n        if not mate_wins(best) and not mate_loses(best):\n            p*=np.exp(-loss/max(25.,cap*.45))\n        if not np.isfinite(p).all() or p.sum()<=0:\n            p=np.zeros(len(candidates));p[int(np.argmax(values))]=1.\n        else:\n            p/=p.sum()\n        error_limit=float(np.interp(profile,[1200,1600],[.08,.03]))\n        if best.get("mate") is None:\n            # Una jugada muy repetida no puede convertir una pérdida apreciable\n            # en la opción habitual: limita su masa conjunta, sin forzar fallos.\n            error_band=loss>60.\n            mass=float(p[error_band].sum())\n            if mass>error_limit:\n                p[error_band]*=error_limit/mass\n                good_mass=float(p[~error_band].sum())\n                if good_mass>0:\n                    p[~error_band]*=(1-error_limit)/good_mass\n                else:\n                    p[int(np.argmax(values))]=1-error_limit\n        return p,dict(safety_cap_cp=cap,blocked_candidates=int((~allowed).sum()),\n                      error_probability_limit=error_limit,\n                      expected_loss_cp=None if best.get("mate") is not None else float(p@loss))\n\n    def _include_anchor(self,rows):\n        anchor=getattr(self,"tactical_anchor",None)\n        if anchor:\n            # No mezcla puntuaciones de búsquedas distintas para la misma jugada.\n            if all(r["move"]!=anchor["move"] for r in rows):\n                anchor=dict(anchor,outside_engine_list=False)\n                rows=([anchor]+rows[:-1]) if rows else [anchor]\n        return sorted(rows,key=lambda r:r["value"],reverse=True)\n\n    def candidates(self,board,profile=1300,remaining=None,maximum=15,use_humans=True):\n        if getattr(self,"anchor_fen",None)!=board.fen():\n            self.tactical_anchor=None\n        if self.prediction_key!=(board.fen(),profile):\n            self.prediction={}\n        if not self.prediction or not use_humans:\n            return self._include_anchor(super().candidates(board,profile,remaining,maximum,use_humans))\n        if not board.is_valid():\n            raise ValueError("Posición inválida.")\n        if board.is_game_over():\n            return []\n        k=min(self.candidate_limit(remaining,maximum),board.legal_moves.count())\n        counts=self.human_counts(board,profile)\n        slots=min(k-1,max(1,k*2//3 if board.ply()<24 else k//2))\n        observed=sorted((m for m,n in counts.items() if n>=2),key=lambda m:-counts[m])\n        predicted=sorted(self.prediction,key=lambda m:-self.prediction[m])\n        # Alterna datos observados y propuestas de la red. Siempre conserva la\n        # primera referencia de Stockfish y vuelve a evaluar el conjunto completo.\n        human=[]\n        for i in range(max(len(observed),len(predicted))):\n            for sequence in (observed,predicted):\n                if i<len(sequence) and sequence[i] not in human:\n                    human.append(sequence[i])\n        human=[chess.Move.from_uci(m) for m in human[:slots]]\n        budget=self.search_seconds(remaining);began=time.monotonic()\n        preliminary=self.engine.analyse(board,chess.engine.Limit(time=budget*.35,nodes=max(100,self.nodes//3)),\n                                        multipv=k,game=self.game_token)\n        references=[x["pv"][0] for x in preliminary if x.get("pv")]\n        if not references:\n            raise RuntimeError("Stockfish no devolvió candidatas.")\n        pool=list(dict.fromkeys([references[0]]+human+references))[:k]\n        infos=self.engine.analyse(board,chess.engine.Limit(time=max(.005,budget-(time.monotonic()-began)),\n                                      nodes=max(100,2*self.nodes//3)),multipv=len(pool),root_moves=pool,game=self.game_token)\n        rows=[self._row(board,info,counts) for info in infos if info.get("pv") and "score" in info]\n        for row in rows:\n            row["outside_engine_list"]=row["move"] not in references\n        return self._include_anchor(rows)\n\n    def distribution(self,board,candidates,profile=1300,beta=1.,temperature=None,remaining=None):\n        base,details=super().distribution(board,candidates,profile,beta,temperature,remaining)\n        details.update(neural_active=False,neural_mix=0.)\n        if (not self.prediction or not self.policy or not self.policy.active or beta<=0\n            or self.prediction_key!=(board.fen(),profile)):\n            base,guard=self.guard_distribution(candidates,base,profile)\n            details.update(guard)\n            return base,details\n        p=np.array([self.prediction.get(row["uci"],0.) for row in candidates],dtype=float)\n        # La red conserva preferencias humanas. El motor aplica una penalización\n        # suave a pérdidas grandes; no convierte su ranking en la etiqueta humana.\n        values=np.array([r["value"] for r in candidates],dtype=float)\n        tolerance=float(np.interp(profile,[1200,1600],[220,100]))*(1.25 if board.ply()<24 else 1.)\n        penalty=np.exp(-np.minimum(700,np.maximum(0,values.max()-values-tolerance)/180))\n        best=candidates[int(np.argmax(values))]\n        for i,row in enumerate(candidates):\n            if best["mate"] is not None and best["mate"]>0 and row["mate"] is None:\n                penalty[i]=math.exp(-max(0,350-(row["cp"] or 0))/200)\n            if row["mate"] is not None and row["mate"]<0 and (best["mate"] is None or best["mate"]>=0):\n                penalty[i]=0\n        p*=penalty\n        if p.sum()>0:\n            p/=p.sum();alpha=min(.5,self.policy.alpha)*min(1.,beta)\n            base=(1-alpha)*base+alpha*p\n            details.update(neural_active=True,neural_mix=alpha)\n        base,guard=self.guard_distribution(candidates,base,profile)\n        details.update(guard)\n        return base,details\n\nclass ReadOnlyMemory:\n    def __init__(self,path):\n        self.conn=sqlite3.connect(\'file:\'+str(Path(path).resolve())+\'?mode=ro\',uri=True)\n        config=json.loads(self.conn.execute("SELECT value FROM meta WHERE key=\'config\'").fetchone()[0])\n        if config[\'schema\']!=2:\n            raise ValueError(\'Se necesita la memoria migrada v2.\')\n        self.rhythm=config[\'rhythm\']\n        self.conn.execute(\'PRAGMA cache_size=-4000\')\n    def counts_for(self,board,profile):\n        pid=jade_position_id(self.conn,pack_position(board))\n        if pid is None:\n            return {}\n        return {unpack_move(code).uci():n for code,n in self.conn.execute(\n            \'SELECT move,n FROM counts WHERE profile=? AND position_id=?\',\n            (rating_band(profile),pid))}\n\ndef main():\n    parser=argparse.ArgumentParser(description=\'Jade en CPU, movimientos UCI y tablero en consola.\')\n    parser.add_argument(\'--stockfish\',required=True,help=\'Ruta al ejecutable de tu dispositivo\')\n    parser.add_argument(\'--memory\',required=True,help=\'Archivo jade2_rapid.sqlite o jade2_blitz.sqlite\')\n    parser.add_argument(\'--model-dir\',default=\'.\')\n    parser.add_argument(\'--fp32\',action=\'store_true\',help=\'Utiliza los pesos originales en lugar de INT8\')\n    parser.add_argument(\'--profile\',type=int,choices=PROFILES,default=1300)\n    parser.add_argument(\'--black\',action=\'store_true\',help=\'Jugar con negras\')\n    args=parser.parse_args();folder=Path(args.model_dir)\n    metadata=json.loads((folder/\'metadata.json\').read_text())\n    precision=\'policy_int8.npz\' if not args.fp32 and metadata[\'quantization\'][\'accepted\'] else \'policy_fp32.npz\'\n    policy=NumpyJadePolicy.load(folder/precision,metadata)\n    memory=ReadOnlyMemory(args.memory)\n    if memory.rhythm!=metadata[\'rhythm\']:\n        raise ValueError(\'El modelo y la memoria son de ritmos distintos.\')\n    engine=JadeEngine(args.stockfish,memory,policy=policy)\n    board=chess.Board();human=not args.black\n    try:\n        while not board.is_game_over():\n            print(board.unicode(borders=True));print(\'Turno:\', \'blancas\' if board.turn else \'negras\')\n            if board.turn==human:\n                text=input(\'Movimiento UCI (o salir): \').strip().lower()\n                if text in (\'salir\',\'quit\',\'exit\'):\n                    return\n                try:\n                    board.push_uci(text)\n                except ValueError:\n                    print(\'Movimiento no legal.\')\n            else:\n                began=time.monotonic();move,detail=engine.choose(board,args.profile)\n                time.sleep(max(0,detail[\'reaction_seconds\']-(time.monotonic()-began)))\n                print(\'Jade:\',board.san(move));board.push(move)\n        print(board);print(board.result())\n    finally:\n        engine.close();memory.conn.close()\nif __name__==\'__main__\':\n    main()\n'
