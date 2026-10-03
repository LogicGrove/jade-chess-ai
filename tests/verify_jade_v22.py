"""Pruebas focalizadas: filtros, aislamiento, reanudación, PGN e integración UCI."""
import hashlib
import json
import random
import tempfile
from pathlib import Path
import chess
import chess.engine
import nbformat
import numpy as np
import torch
from threadpoolctl import threadpool_limits
import sys
REPO_ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO_ROOT/'src'))
import Jade_Programa as j

ROOT=Path(tempfile.mkdtemp(prefix="jade22-check-"))
threadpool_limits(1);torch.set_num_threads(1)
memory=j.JadeMemory(ROOT/"data.sqlite")
rng=random.Random(62)
with memory.conn:
    for i in range(50):
        board=chess.Board();codes=[]
        for ply in range(20):
            if board.is_game_over(): break
            move=rng.choice(list(board.legal_moves));codes.append(j.pack_move(move));board.push(move)
        record=dict(moves=codes,clocks=[500]*len(codes),ratings=[1300,1300],time_control="600+0",rhythm="rapid")
        key=hashlib.sha256(np.array(codes,dtype="<u2").tobytes()).hexdigest()
        split="train" if i<20 else "validation" if i<40 else "test"
        payload=j.zstd.ZstdCompressor().compress(json.dumps(record).encode())
        memory.conn.execute("INSERT INTO corpus(game_id,fingerprint,split,payload) VALUES(?,?,?,?)",(str(i),key,split,payload))
before=list(memory.conn.iterdump())

assert j.jade_quality_weight(chess.engine.Cp(10),chess.engine.Cp(-290))[0]==0
assert j.jade_quality_weight(chess.engine.Cp(10),chess.engine.Cp(-50))[0]==.6
assert j.jade_quality_weight(chess.engine.Cp(10),chess.engine.Mate(-3))[0]==0
assert j.jade_quality_weight(chess.engine.Mate(2),chess.engine.Cp(900))[0]==0
assert j.jade_quality_weight(chess.engine.Mate(2),chess.engine.Mate(7))[0]==1

def row(value,mate=None): return dict(value=value,mate=mate,cp=value if mate is None else None)
for profile in j.PROFILES:
    p,d=j.JadeEngine.guard_distribution([row(50),row(-20),row(-400)],np.array([.0001,.0001,.9998]),profile)
    assert p[2]==0 and np.isclose(p.sum(),1)
    p,d=j.JadeEngine.guard_distribution([row(0),row(-65)],np.array([.001,.999]),profile)
    assert p[1]<=d["error_probability_limit"]+1e-10
    p,d=j.JadeEngine.guard_distribution([row(99997,3),row(500),row(-99999,-1)],[.001,.998,.001],profile)
    assert np.array_equal(p,[1,0,0])
    p,d=j.JadeEngine.guard_distribution([row(-99997,-3),row(-99999,-1)],[.4,.6],profile)
    assert np.isfinite(p).all() and np.isclose(p.sum(),1)
print("PASS: límites, errores graves, mates y normalización.")

quality=j.open_jade_quality(ROOT/"local",ROOT/"models","rapid")
for seq,key,split,payload in memory.conn.execute("SELECT seq,fingerprint,split,payload FROM corpus WHERE split!='test'"):
    labels={}
    for x in j.jade_record_examples(payload,0,key,4):
        labels[str(x["ply"])]=dict(uci=x["uci"],weight=0. if len(labels)==0 else .25 if len(labels)==1 else 1.)
    with quality.conn:
        quality.conn.execute("INSERT INTO reviewed VALUES(?,?,?,?)",(key,seq,split,json.dumps(labels)))
examples=list(quality.examples(memory,"train",max_games=20))
assert examples and all(x["weight"]>0 for x in examples)
assert all("board" not in x for x in examples)
try:
    list(quality.examples(memory,"test"))
    raise AssertionError("Se aceptó test para el ajuste")
except ValueError: pass
quality.save();quality.close()
restored=j.open_jade_quality(ROOT/"local2",ROOT/"models","rapid")
assert restored.summary()=={"train":20,"validation":20}
model_dir=ROOT/"models"
j.train_jade_policy(memory,model_dir,epochs=1,max_games=3,positions_per_game=4,
                    device="cpu",validation_games=3,log=lambda s:None)
checkpoint=model_dir/"training.pt";digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
old_state=torch.load(checkpoint,weights_only=True)["state"]
result=j.train_jade_policy(memory,model_dir/"calidad_22",epochs=1,max_games=8,
     quality=restored,initial_checkpoint=checkpoint,device="cpu",validation_games=20,log=lambda s:None)
assert result["metadata"]["validation"]["objective"]=="human_quality_weighted"
assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==digest
saved=torch.load(model_dir/"calidad_22"/"training.pt",weights_only=True)
assert saved["state"]["steps"]>old_state["steps"]
result2=j.train_jade_policy(memory,model_dir/"calidad_22",epochs=1,max_games=8,
     quality=restored,initial_checkpoint=checkpoint,device="cpu",validation_games=20,log=lambda s:None)
assert torch.load(model_dir/"calidad_22"/"training.pt",weights_only=True)["state"]["steps"]>saved["state"]["steps"]
policy=j.NumpyJadePolicy(j.numpy_jade_weights(j.make_jade_network()))
raw=j.evaluate_jade_policy(policy,memory,split="test",max_games=5)
assert raw["objective"]=="human_imitation" and "weight_sum" not in raw
assert list(memory.conn.iterdump())==before
print("PASS: corpus intacto, separación train/validation/test, respaldo, pesos originales intactos y reanudación CPU.")

nb=nbformat.read("Jade_Colab.ipynb",as_version=4);nbformat.validate(nb)
for cell in nb.cells:
    if cell.cell_type=="code" and not any(line.startswith(("%","!")) for line in cell.source.splitlines()):
        compile(cell.source,"cell","exec")
assert any("#@title 9A." in c.source for c in nb.cells)
print("PASS: notebook autónomo y sintaxis de celdas.")

binary=Path("stockfish/stockfish-ubuntu-x86-64-avx2")
if binary.exists():
    engine=j.JadeEngine(binary.resolve(),memory,nodes=12000)
    for board in [chess.Board("7k/8/5KQ1/8/8/8/8/8 w - - 0 1"),chess.Board("7k/8/5KQ1/8/8/8/8/8 w - - 0 1").mirror()]:
        for profile in (1200,1600):
            move,details=engine.choose(board,profile,pace=0)
            after=board.copy();after.push(move)
            assert after.is_checkmate(),(board.fen(),move,details)
    engine.close()
    fresh=j.open_jade_quality(ROOT/"fresh",ROOT/"freshmodels","rapid")
    j.prepare_jade_quality(memory,fresh,binary.resolve(),train_games=2,validation_games=2,
                          positions_per_game=2,seconds=.02,nodes=2000,log=lambda s:None)
    assert fresh.summary()=={"train":2,"validation":2}
    j.prepare_jade_quality(memory,fresh,binary.resolve(),train_games=2,validation_games=2,
                          positions_per_game=2,seconds=.02,nodes=2000,log=lambda s:None)
    assert fresh.summary()=={"train":4,"validation":2}
    fresh.close()
    report=j.analizar_pgn_jade('[White "Prueba"]\n[Black "Jade"]\n\n1. f3 e5 2. g4 Qh4# 0-1',
                              binary.resolve(),ROOT/"reports",seconds=.03,threads=1,hash_mb=16,log=lambda s:None)
    assert report["metadata"]["complete"] and report["rows"][-1]["played_score"]["value"]<0
    assert report["rows"][-1]["played_score"]["mate"]==0
    Path("qa_report.html").write_text(report["html"].read_text())
    print("PASS: Stockfish real, mate en ambos colores, revisión incremental y PGN con mate negro.")
else:
    print("PENDIENTE: integración con un ejecutable real de Stockfish.")
restored.close();memory.close()
print("TEMP:",ROOT)
