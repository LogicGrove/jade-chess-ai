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
