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
