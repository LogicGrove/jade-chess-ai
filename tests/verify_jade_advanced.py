"""Verifica revisión previa, pesos fraccionarios y reanudación desde copias."""
import hashlib
import json
import random
import tempfile
from pathlib import Path
from unittest.mock import patch

import chess
import chess.pgn
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import Jade_Programa as j

root=Path(tempfile.mkdtemp(prefix="jade-advanced-check-"))
rng=random.Random(82);by_split={s:[] for s in ("train","validation","test")}
for number in range(300):
    game=chess.pgn.Game();node=game;board=game.board();codes=[]
    for ply in range(10):
        if board.is_game_over():
            break
        move=rng.choice(list(board.legal_moves));codes.append(j.pack_move(move))
        node=node.add_variation(move);board.push(move)
    key=hashlib.sha256(np.array(codes,dtype="<u2").tobytes()).hexdigest()
    row=dict(Event=j.EVENTS["rapid"],Site=f"local/{number}",WhiteElo=1300,BlackElo=1400,
             TimeControl="600+0",movetext=str(game))
    by_split[j.jade_split(key)].append((row,key))
rows=[by_split["train"][0],by_split["train"][1],by_split["validation"][0],by_split["test"][0]]
parquet=root/"games.parquet"
pq.write_table(pa.Table.from_pylist([x[0] for x in rows]+[rows[0][0]]),parquet,row_group_size=5)
entries=[dict(path=str(parquet))]
memory=j.JadeMemory(root/"data.sqlite");quality=j.open_jade_quality(root,root/"models","rapid")
calls=[];stop_at=[13];settings=[]

class FakeEngine:
    id={"name":"Motor de prueba"}
    def configure(self,config): settings.append(config)
    def quit(self): pass

def fake_review(engine,board,move,seconds,nodes,token):
    if len(calls)==stop_at[0]:
        raise KeyboardInterrupt()
    calls.append((tuple(m.uci() for m in board.move_stack),move.uci()))
    values={0:(.6,60),1:(.25,100),2:(0.,300),3:(.05,200)}
    weight,loss=values.get(board.ply(),(1.,0))
    return dict(uci=move.uci(),weight=weight,loss_cp=loss,kind="cp")

with patch.object(j.chess.engine.SimpleEngine,"popen_uci",return_value=FakeEngine()),patch.object(j,"jade_review_move",fake_review):
    review=j.JadeAdvancedReviewer(quality,"fake",error_fraction=0,log=lambda _:None)
    report=j.feed_jade(memory,new_games=20,max_rows=20,file_entries=entries,backup_dir=root/"backups",
                       quality_reviewer=review,log=lambda _:None)
    review.close()
    assert report["nueva"]==1
    assert memory.conn.execute("SELECT COUNT(*) FROM corpus").fetchone()[0]==1
    assert memory.conn.execute("SELECT COUNT(*) FROM quality_labels").fetchone()[0]==1
    assert quality.conn.execute("SELECT complete FROM advanced_review").fetchall()==[(0,)]
    assert all(x["Skill Level"]==20 and x["UCI_LimitStrength"] is False for x in settings)
    prior_calls=list(calls);memory.close();quality.close()

    # Simula pérdida del entorno local: solo se usan las copias comprimidas.
    j.restore_memory(root/"recovered"/"data.sqlite",root/"backups","rapid")
    memory=j.JadeMemory(root/"recovered"/"data.sqlite")
    quality=j.open_jade_quality(root/"recovered",root/"models","rapid")
    stop_at[0]=-1
    review=j.JadeAdvancedReviewer(quality,"fake",error_fraction=0,log=lambda _:None)
    report=j.feed_jade(memory,new_games=20,max_rows=20,file_entries=entries,backup_dir=root/"backups",
                       quality_reviewer=review,log=lambda _:None)
    review.close()
    assert report["nueva"]==3 and report["duplicada"]==1
    # Las jugadas confirmadas de la partida interrumpida no se vuelven a analizar.
    assert all(c not in calls[len(prior_calls):] for c in prior_calls[10:])
    assert memory.conn.execute("SELECT COUNT(*) FROM corpus").fetchone()[0]==4
    assert memory.conn.execute("SELECT COUNT(*) FROM quality_labels").fetchone()[0]==3
    assert quality.summary()=={"train":2,"validation":1}
    assert quality.conn.execute("SELECT COUNT(*) FROM advanced_review").fetchone()[0]==0

expected={}
for seq,key,split,packed in memory.conn.execute("SELECT seq,fingerprint,split,payload FROM corpus"):
    labels=memory.conn.execute("SELECT labels FROM quality_labels WHERE seq=?",(seq,)).fetchone()
    if split=="test":
        assert labels is None
        continue
    labels=json.loads(labels[0]);record=json.loads(j.zstd.ZstdDecompressor().decompress(packed));board=chess.Board()
    assert len(labels)==len(record["moves"])
    for ply,code in enumerate(record["moves"]):
        if split=="train" and labels[str(ply)]["weight"]>0:
            profile=j.rating_band(record["ratings"][int(board.turn)])
            key=(profile,j.pack_position(board),code)
            expected[key]=expected.get(key,0.)+labels[str(ply)]["weight"]
        board.push(j.unpack_move(code))
actual={(p,key,m):n for p,key,m,n in memory.conn.execute(
    "SELECT c.profile,p.key,c.move,c.n FROM counts c JOIN positions p ON p.id=c.position_id")}
assert actual==expected
dummy=object.__new__(j.JadeEngine);dummy.memory=memory
assert any(not float(x).is_integer() for x in dummy.human_counts(chess.Board(),1300).values())
examples=list(quality.examples(memory,"train"))
assert examples and all(x["weight"]>0 for x in examples) and any(x["weight"]==.6 for x in examples)
assert all(x["ply"]!=2 for x in examples)

# Si se pierde solo la copia auxiliar, las etiquetas confirmadas siguen en memoria.
q2=j.open_jade_quality(root/"fresh",root/"freshmodels","rapid")
assert j.sync_jade_quality(memory,q2)==3 and j.sync_jade_quality(memory,q2)==0
assert q2.summary()==quality.summary()

# La selección rara es fija, pequeña, separada de validación y de errores de mate.
bad=dict(uci="a2a3",weight=0.,loss_cp=400,kind="cp")
selected=[j.jade_rare_error_weight(bad,str(i),4,"train",.02) for i in range(10000)]
count=sum(x["rare_error"] for x in selected)
assert 150<count<250
for i in range(100):
    assert j.jade_rare_error_weight(bad,str(i),4,"train",.02)==selected[i]
    assert j.jade_rare_error_weight(bad,str(i),4,"validation",.05)["weight"]==0
    assert j.jade_rare_error_weight(dict(bad,kind="permite_mate"),str(i),4,"train",.05)["weight"]==0
assert all(x["weight"] in (0.,.05) for x in selected)
print("PASS: revisión antes de incorporar, pesos, test intacto, duplicados, interrupción y restauración de ambas bases.")
print("PASS: muestra determinista de errores:",count,"/ 10000; validación y mates excluidos.")

# El modo normal conserva el peso unitario y no genera etiquetas.
normal=j.JadeMemory(root/"normal.sqlite")
j.feed_jade(normal,new_games=10,max_rows=10,file_entries=entries,backup_dir=root/"normal_backups",log=lambda _:None)
assert normal.conn.execute("SELECT COUNT(*) FROM quality_labels").fetchone()[0]==0
assert normal.conn.execute("SELECT SUM(n) FROM counts").fetchone()[0]==20
normal.close()

binary=Path("stockfish/stockfish-ubuntu-x86-64-avx2")
if binary.exists():
    real=j.JadeAdvancedReviewer(q2,binary.resolve(),seconds=.03,nodes=3000,log=lambda _:None)
    board=chess.Board()
    for uci in ("f2f3","e7e5","g2g4"):
        board.push_uci(uci)
    label=j.jade_review_move(real.engine,board,chess.Move.from_uci("d8h4"),.03,3000,object())
    assert label["weight"]==1 and label["kind"]=="mantiene_mate"
    label=j.jade_review_move(real.engine,board,chess.Move.from_uci("b8c6"),.03,3000,object())
    assert label["weight"]==0 and label["kind"]=="omite_mate"
    real.close()
    print("PASS: comparación UCI real desde negras, mate conservado y mate omitido.")
q2.close();quality.close();memory.close()
