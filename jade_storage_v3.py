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
