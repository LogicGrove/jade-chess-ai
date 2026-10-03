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
