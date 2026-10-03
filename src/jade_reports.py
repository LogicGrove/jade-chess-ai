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
