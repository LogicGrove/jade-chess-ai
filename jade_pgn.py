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
