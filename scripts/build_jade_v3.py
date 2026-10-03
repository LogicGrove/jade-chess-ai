"""Construye el cuaderno autónomo y el programa ligero a partir de las fuentes."""
import ast
import json
from pathlib import Path
from textwrap import dedent

REPO_ROOT=Path(__file__).resolve().parents[1]
SOURCE_DIR=REPO_ROOT/'src'
base=json.loads((REPO_ROOT/'scripts'/'templates'/'Jade_v12_base.ipynb').read_text())
def source(cell):
    return ''.join(cell['source']) if isinstance(cell['source'],list) else cell['source']
def selected(path,names):
    text=(SOURCE_DIR/path).read_text();tree=ast.parse(text)
    return '\n\n'.join(ast.get_source_segment(text,node) for node in tree.body if getattr(node,'name',None) in names)

light='''"""Jade 2.2.1 en CPU: requiere numpy, python-chess y un Stockfish local."""
import argparse, hashlib, json, math, sqlite3, time
from pathlib import Path
import numpy as np
import chess, chess.engine
PROFILES=(1200,1300,1400,1500,1600)
JADE_FEATURES=1560
def rating_band(rating):
    return 100*math.floor((int(rating)+50)/100)
'''
light+=selected('jade_storage_v3.py',{'pack_position','unpack_move','jade_position_id'})+'\n'
light+=selected('jade_policy.py',{'jade_context','encode_jade','NumpyJadePolicy'})+'\n'
light+=(SOURCE_DIR/'jade_engine_v2.py').read_text()+'\nClassicJadeEngine=JadeEngine\n'
light+=(SOURCE_DIR/'jade_hybrid_v3.py').read_text()
light+='''
class ReadOnlyMemory:
    def __init__(self,path):
        self.conn=sqlite3.connect('file:'+str(Path(path).resolve())+'?mode=ro',uri=True)
        config=json.loads(self.conn.execute("SELECT value FROM meta WHERE key='config'").fetchone()[0])
        if config['schema']!=2:
            raise ValueError('Se necesita la memoria migrada v2.')
        self.rhythm=config['rhythm']
        self.conn.execute('PRAGMA cache_size=-4000')
    def counts_for(self,board,profile):
        pid=jade_position_id(self.conn,pack_position(board))
        if pid is None:
            return {}
        return {unpack_move(code).uci():n for code,n in self.conn.execute(
            'SELECT move,n FROM counts WHERE profile=? AND position_id=?',
            (rating_band(profile),pid))}

def main():
    parser=argparse.ArgumentParser(description='Jade en CPU, movimientos UCI y tablero en consola.')
    parser.add_argument('--stockfish',required=True,help='Ruta al ejecutable de tu dispositivo')
    parser.add_argument('--memory',required=True,help='Archivo jade2_rapid.sqlite o jade2_blitz.sqlite')
    parser.add_argument('--model-dir',default='.')
    parser.add_argument('--fp32',action='store_true',help='Utiliza los pesos originales en lugar de INT8')
    parser.add_argument('--profile',type=int,choices=PROFILES,default=1300)
    parser.add_argument('--black',action='store_true',help='Jugar con negras')
    args=parser.parse_args();folder=Path(args.model_dir)
    metadata=json.loads((folder/'metadata.json').read_text())
    precision='policy_int8.npz' if not args.fp32 and metadata['quantization']['accepted'] else 'policy_fp32.npz'
    policy=NumpyJadePolicy.load(folder/precision,metadata)
    memory=ReadOnlyMemory(args.memory)
    if memory.rhythm!=metadata['rhythm']:
        raise ValueError('El modelo y la memoria son de ritmos distintos.')
    engine=JadeEngine(args.stockfish,memory,policy=policy)
    board=chess.Board();human=not args.black
    try:
        while not board.is_game_over():
            print(board.unicode(borders=True));print('Turno:', 'blancas' if board.turn else 'negras')
            if board.turn==human:
                text=input('Movimiento UCI (o salir): ').strip().lower()
                if text in ('salir','quit','exit'):
                    return
                try:
                    board.push_uci(text)
                except ValueError:
                    print('Movimiento no legal.')
            else:
                began=time.monotonic();move,detail=engine.choose(board,args.profile)
                time.sleep(max(0,detail['reaction_seconds']-(time.monotonic()-began)))
                print('Jade:',board.san(move));board.push(move)
        print(board);print(board.result())
    finally:
        engine.close();memory.conn.close()
if __name__=='__main__':
    main()
'''
(SOURCE_DIR/'Jade_Ligero.py').write_text(light)

core=(SOURCE_DIR/'jade_core.py').read_text().split('\nclass JadeEngine:',1)[0]
core=core.replace('Jade 1.0: Stockfish MultiPV + Boltzmann + frecuencias humanas persistentes.',
                  'Jade 2.2.1: predictor humano compacto, enseñanza avanzada y memoria persistente.')
core=core.replace('("WhiteTitle", "BlackTitle")','("WhiteTitle", "BlackTitle", "TimeControl")')
core+='\nLegacyJadeMemory=JadeMemory\nlegacy_restore_memory=restore_memory\nlegacy_save_memory=save_memory\n'
core+='\n'+(SOURCE_DIR/'jade_storage_v3.py').read_text()+'\n'+(SOURCE_DIR/'jade_policy.py').read_text()
core+='\n'+(SOURCE_DIR/'jade_quality.py').read_text()
core+='\n'+(SOURCE_DIR/'jade_advanced.py').read_text()
core+='\n'+(SOURCE_DIR/'jade_reports.py').read_text()
core+='\n'+(SOURCE_DIR/'jade_engine_v2.py').read_text()+'\nClassicJadeEngine=JadeEngine\n'
core+='\n'+(SOURCE_DIR/'jade_hybrid_v3.py').read_text()+'\n'+(SOURCE_DIR/'jade_ui.py').read_text()+'\n'+(SOURCE_DIR/'jade_pgn.py').read_text()
core+='''
def memory_report(memory):
    report=memory.summary()
    print(f"Jade 2.2.1 · {memory.rhythm}: {report['partidas']:,} partidas registradas; {report['base_MB']:.2f} MB locales")
    print('Corpus nuevo:',report['corpus'],'| Antiguas conservadas:',report['legacy_games'])
    print('Perfil | Partidas que alimentan frecuencias | Movimientos')
    for p in PROFILES:
        s=report['perfiles'].get(p,{'partidas':0,'movimientos':0})
        print(p,s['partidas'],s['movimientos'])
    return report
'''
core+='\nJADE_LIGHT_SOURCE='+repr(light)+'\n'
(SOURCE_DIR/'Jade_Programa.py').write_text(core)

cells=[]
def md(text):
    cells.append(dict(cell_type='markdown',metadata={},source=dedent(text).strip()+'\n'))
def code(text,hidden=False):
    cells.append(dict(cell_type='code',metadata={'cellView':'form'} if hidden else {},execution_count=None,outputs=[],source=dedent(text).strip()+'\n'))
def old_code(number):
    return next(source(c) for c in base['cells'] if c['cell_type']=='code' and source(c).startswith(f'#@title {number}.'))

md('''
# Jade 2.2.1
Control de errores graves, ajuste con partidas revisadas y barra visual de evaluación.
Compatible con memorias y modelos 2.0/2.1. No hace falta repetir la ingesta ni convertir otra vez.

**Para probar la corrección inmediata:** guarda la memoria del cuaderno anterior con 7,
detén aquel cuaderno y ejecuta aquí **1 → 2 → 3 → 3B → 4 → 6** con el mismo ritmo/cuenta.
El filtro táctico actúa sobre tu modelo existente. No certifica el ELO seleccionado.
**Para corregir también la red:** ejecuta **9A → 9**. Revisa una muestra y conserva
el modelo original. El modelo de calidad se activa solo si supera su validación ponderada.
**Para ver la barra:** calcula un informe nuevo en 8; los HTML antiguos no cambian solos.

**Nuevo: enseñanza avanzada en 5.** Activa la casilla, empieza con 100 partidas y
entrena después en 9 con revisión de calidad. Analiza antes de sumar frecuencias,
da menos peso a fallos y permite conservar una muestra pequeña de errores graves.
Las partidas antiguas y los modelos se conservan. 9A sigue siendo una alternativa
para revisar una muestra de las partidas ya incorporadas.

**Si ya usas 2.0:** guarda con 7 y abre este cuaderno con la misma cuenta, carpeta y
ritmo. Ejecuta **1 → 2 → 3 → 3B → 4**. Recupera la memoria y los pesos existentes;
no tienes que volver a descargar partidas ni empezar a entrenar de cero.

**Actualización desde 1.x:** guarda con la celda 7 del cuaderno antiguo si sigue conectado.
Aquí ejecuta **1 → 2 → 3 → 3B → 4** usando la misma cuenta y el mismo ritmo. La celda
3B convierte la memoria a otro archivo, verifica cada frecuencia y conserva el original.
Usa un único cuaderno activo por memoria.

Después: **6 jugar**, **5 añadir partidas**, **9 entrenar**, **10 examinar**, **11 exportar
para CPU**. El análisis PGN permanece en **8** y los informes guardados se reabren en **8B**.

En una instalación nueva, la red empieza sin entrenar. Migrar conserva las frecuencias, no inventa pesos neuronales.
El preentrenamiento opcional de 9 aprovecha esas estadísticas; las partidas nuevas aportan
secuencias completas y exámenes independientes. El rival conserva su funcionamiento si
la red no está aprobada.

Para jugar, CPU y ninguna VRAM para el predictor. Para entrenar, T4 opcional o CPU;
la velocidad real depende también de preparar las partidas. No se garantiza un
ELO ni calidad idéntica en todos los dispositivos. No se incluye un modelo preentrenado.
''')
md('## 1 · Preparar el entorno\nNo reinstala PyTorch. Se utiliza el que trae Colab al entrenar; jugar solo necesita NumPy para la red.')
install=old_code(1).replace(' fsspec',' fsspec "zstandard>=0.23,<1" threadpoolctl')
code(install+'\nfrom threadpoolctl import threadpool_limits\nlimite_blas_jade=threadpool_limits(limits=1)\n')
md('## 2 · Datos y almacenamiento\nEl año/mes selecciona la fuente de partidas. Rapid y blitz conservan memorias y modelos independientes.')
code('''
#@title 2. Ajustes de Jade
GUARDAR_EN_DRIVE=True #@param {type:"boolean"}
RITMO="rapid" #@param ["rapid","blitz"]
ANO=2024 #@param {type:"integer"}
MES=1 #@param {type:"integer"}
NUEVAS_PARTIDAS=2000 #@param {type:"integer"}
MAX_FILAS=50000 #@param {type:"integer"}
MAX_ARCHIVOS=3 #@param {type:"integer"}
NODOS_STOCKFISH=100000 #@param {type:"integer"}
HASH_STOCKFISH_MB=64 #@param [32,64,128] {type:"raw"}
if RITMO not in ("rapid","blitz") or not 1<=MES<=12 or min(NUEVAS_PARTIDAS,MAX_FILAS,MAX_ARCHIVOS,NODOS_STOCKFISH)<1:
    raise ValueError("Revisa los límites y el mes.")
LOCAL_DIR=Path("/content/Jade");LOCAL_DIR.mkdir(parents=True,exist_ok=True)
BASE_DIR=LOCAL_DIR
if GUARDAR_EN_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    BASE_DIR=Path("/content/drive/MyDrive/Jade")
BASE_DIR.mkdir(parents=True,exist_ok=True)
LEGACY_BACKUP_DIR=BASE_DIR/"copias"
BACKUP_DIR=BASE_DIR/"copias_v2" if GUARDAR_EN_DRIVE else None
MODEL_DIR=BASE_DIR/"modelos"/RITMO
DB_PATH=LOCAL_DIR/f"jade2_{RITMO}.sqlite"
OLD_DB_PATH=LOCAL_DIR/f"jade_{RITMO}.sqlite"
print("Ritmo:",RITMO,"| Modelos:",MODEL_DIR)
''')
md('## 3 · Programa completo\nEjecuta la celda plegada. No hace falta subir ningún archivo .py aparte.')
code('#@title 3. Cargar Jade 2.2.1\n'+core,hidden=True)
md('''
## 3B · Migrar o recuperar
Recupera primero una copia v2. Si no existe, convierte la antigua a otro archivo y
verifica cada frecuencia, partidas, perfiles y cursores. No borra el original.
Repetir la celda conserva una base nueva existente y no vuelve a importar ni duplicar.

La clave de posición usa 34 bytes, más índices y referencias; se comparte entre perfiles.
Las copias v2 usan Zstandard y las antiguas gzip permanecen intactas. Las estadísticas
antiguas no contienen la secuencia completa ni los relojes: se conservan como aprendizaje,
nunca como examen independiente.
''')
code('''
#@title 3B. Convertir y verificar la memoria antigua
if "JadeMemory" not in globals() or "DB_PATH" not in globals():
    raise RuntimeError("Ejecuta antes 1, 2 y 3.")
if "tablero_jade" in globals():
    tablero_jade.close()
if "jade" in globals():
    jade.close()
if "memoria" in globals():
    memoria.close()
restore_memory(DB_PATH,BACKUP_DIR,RITMO)
if not DB_PATH.exists():
    legacy_restore_memory(OLD_DB_PATH,LEGACY_BACKUP_DIR,RITMO)
    if OLD_DB_PATH.exists():
        migrate_jade_memory(OLD_DB_PATH,DB_PATH,RITMO)
    else:
        nueva=JadeMemory(DB_PATH,RITMO);nueva.close()
        print("No hay copia antigua. Memoria nueva vacía preparada.")
comprobacion=JadeMemory(DB_PATH,RITMO)
try:
    memory_report(comprobacion)
    save_memory(comprobacion,BACKUP_DIR)
finally:
    comprobacion.close()
print("Siguiente paso: celda 4.")
''')
md('## 4 · Iniciar Jade\nCarga un predictor aprobado si existe. Un modelo aleatorio no modifica sus decisiones.')
code('''
#@title 4. Abrir memoria y motor
if "JadeMemory" not in globals() or not DB_PATH.exists():
    raise RuntimeError("Ejecuta 1, 2, 3 y 3B.")
if "tablero_jade" in globals():
    tablero_jade.close()
if "jade" in globals():
    jade.close()
if "memoria" in globals():
    memoria.close()
memoria=JadeMemory(DB_PATH,RITMO)
predictor=load_jade_play_policy(MODEL_DIR,RITMO)
binary=shutil.which("stockfish") or "/usr/games/stockfish"
jade=JadeEngine(binary,memoria,nodes=NODOS_STOCKFISH,policy=predictor,hash_mb=HASH_STOCKFISH_MB)
memory_report(memoria)
print("Motor:",jade.engine.id.get("name"),"| Predictor aprobado:",bool(predictor and predictor.active))
print("Jade 2.2.1: protección táctica activa. Predictor:",predictor.metadata.get("objective","humano original") if predictor else "solo frecuencias")
''')
md('''
## 5 · Incorporar otro lote
Lee Parquet por bloques remotos, conserva el cursor antiguo y descarta IDs ya vistos.
Las nuevas secuencias se separan aproximadamente 80 % entrenamiento, 10 % validación
y 10 % test según una huella de movimientos. Secuencias idénticas no cruzan grupos.
Validación y test no alimentan ni las frecuencias ni la red. No garantiza jugadores
distintos y las aperturas comunes pueden aparecer en varios grupos.

Se guardan las primeras 100 medias jugadas, ratings y relojes si existen, sin nombres
de usuario. «Partidas registradas» incluye las reservadas. Las partidas contra Jade no
se incorporan. Esta celda añade datos; la red se entrena en 9.

**Nuevo en 2.2.1: MODO_ENSENANZA_AVANZADA.** Activa la casilla para revisar todos los
movimientos del tramo guardado que pertenezcan a los perfiles de Jade, antes de sumar
frecuencias. Stockfish usa fuerza completa y búsquedas breves; el ELO humano se conserva
como contexto. El test queda sin filtrar y no se analiza para aprender. Las jugadas
forzadas reciben peso 1 sin gastar una búsqueda.

Empieza con PARTIDAS_AVANZADAS=100 y 0,08 s por búsqueda. Es un máximo de partidas
aceptadas de este lote, además del límite de la celda 2. Puede tardar muchos minutos;
cada posición necesita hasta tres búsquedas. Stockfish usa CPU, no la T4.
Se guarda por partida; una revisión interrumpida reutiliza movimientos ya analizados.

ERRORES_GRAVES_CONSERVADOS_PCT=2 selecciona aproximadamente el 2 % de los fallos
de más de 250 cp de entrenamiento, con peso 0,05. El resto pesa cero. No se aplica a
errores de mate ni a validación; no significa que Jade vaya a fallar un 2 % al jugar.
Es una selección fija, no otro sorteo en cada época. Pon 0 para excluirlos todos.
Los parámetros nuevos se aplican a partidas todavía no revisadas. Las antiguas no
se reponderan solas; 9A sigue disponible para revisar una muestra de ellas.

Después ejecuta **9 con USAR_REVISION_CALIDAD=True**. No hace falta pasar por 9A si
ya hay entrenamiento y validación revisados suficientes. La red no cambia solo por
ejecutar 5; las nuevas frecuencias sí quedan ponderadas. Desactivar el modo devuelve
la incorporación normal, y entrenar sin revisión utiliza de nuevo las etiquetas crudas.
''')
code('''
#@title 5. Añadir partidas nuevas y guardar
MODO_ENSENANZA_AVANZADA=False #@param {type:"boolean"}
PARTIDAS_AVANZADAS=100 #@param {type:"integer"}
SEGUNDOS_ANALISIS_AVANZADO=0.08 #@param {type:"number"}
ERRORES_GRAVES_CONSERVADOS_PCT=2 #@param {type:"slider",min:0,max:5,step:1}
if "memoria" not in globals() or "JadeAdvancedReviewer" not in globals():
    raise RuntimeError("Ejecuta primero 1, 2, 3, 3B y 4 del cuaderno actualizado.")
if memoria.rhythm != RITMO:
    raise ValueError("Has cambiado de ritmo: ejecuta primero la celda 4.")
if MODO_ENSENANZA_AVANZADA and PARTIDAS_AVANZADAS<1:
    raise ValueError("PARTIDAS_AVANZADAS debe ser positivo.")
revision=None;revisor=None
try:
    if MODO_ENSENANZA_AVANZADA:
        if "tablero_jade" in globals():
            tablero_jade.settle_clock();tablero_jade.paused=True;tablero_jade.clock_started=None;tablero_jade.save_game()
        revision=open_jade_quality(DB_PATH.parent,MODEL_DIR,RITMO)
        sync_jade_quality(memoria,revision)
        revisor=JadeAdvancedReviewer(revision,jade.binary,seconds=SEGUNDOS_ANALISIS_AVANZADO,
                                     error_fraction=ERRORES_GRAVES_CONSERVADOS_PCT/100)
        print("Modo avanzado: revisión antes de incorporar; test intacto. Luego entrena en 9.")
    resultado_carga=feed_jade(memoria,year=ANO,month=MES,
        new_games=min(NUEVAS_PARTIDAS,PARTIDAS_AVANZADAS) if MODO_ENSENANZA_AVANZADA else NUEVAS_PARTIDAS,
        max_rows=MAX_FILAS,max_files=MAX_ARCHIVOS,backup_dir=BACKUP_DIR,quality_reviewer=revisor)
    print("Resultado de este lote:",resultado_carga)
    memory_report(memoria)
finally:
    try:
        if revisor is not None:
            revisor.close()
    finally:
        if revision is not None:
            revision.close()
''',hidden=True)
md('''
## 6 · Jugar
Elige color, perfil y reloj. Toca origen/destino. El dibujo usa SVG de python-chess.
Las opciones máximas según el tiempo de Jade son 15/12/8/5/3, limitadas por las legales.
La red aprobada puede proponer jugadas fuera del ranking inicial de Stockfish.
En 2.2 todas las propuestas pasan por el mismo control: límite heurístico de
pérdida y conservación de mates detectados. Se confirma la elección frente a
una referencia táctica. Puede aumentar el cálculo hasta ~1,8 veces el presupuesto
anterior; el reloj descuenta ese tiempo. No hay garantía de detectar todas las tácticas.

La respuesta Natural suele tardar 2–9 s con tiempo abundante y menos en apuros.
El reloj descuenta cálculo y pausa. Puedes pausar/deshacer. Una partida recuperada se
abre en pausa; una partida antigua sin reloj sigue sin reloj hasta iniciar otra.
''')
code(old_code(6).replace('(Path(BACKUP_DIR).parent if BACKUP_DIR is not None else Path(DB_PATH).parent)','BASE_DIR'))
md('## 7 · Guardar\nTres copias v2 por ritmo, además de las antiguas. Compresión sin pérdida, sin descartar posiciones raras. Colab utiliza la base descomprimida.')
code(old_code(7))
md('''
## 8 · Analizar PGN con Stockfish
Sin humanización, máxima fuerza configurada y tiempo finito por búsqueda. Exporta HTML,
PGN y CSV. Las etiquetas son propias y orientativas, y las evaluaciones se muestran
desde blancas. El visor se abre en un iframe independiente para evitar el fallo de la
salida de widgets. Si ya has analizado, ejecuta 8B y no repitas el cálculo.
''')
code(old_code(8).replace('de Jade 1.2','de Jade 2.0'))
code('''
#@title 8B. Abrir el último informe ya calculado
abrir_ultimo_informe_jade(BASE_DIR/"analisis")
''')
md('''
## 9A · Revisar una muestra para aprender con menos errores
Stockfish compara la jugada humana con una referencia desde el punto de vista
del jugador que mueve. Mantiene peso 1 hasta 40 cp; aplica 0,6 hasta 80 cp,
0,25 hasta 140 cp, 0,05 hasta 250 cp y 0 por encima o al permitir/omitir mate.
Son umbrales heurísticos, no medidas exactas del valor de un gambito ni de ELO.
Los casos de peso cero se comprueban con más cálculo antes de excluirlos.

Empieza con **250 partidas de entrenamiento, 150 de validación y 6 posiciones por
partida**. Puede tardar varios minutos, según CPU. Repetir añade otro lote de
entrenamiento; reutiliza la validación y las etiquetas guardadas. No analiza los
dos millones de partidas. Las antiguas frecuencias sin secuencias se conservan,
pero no sirven para este ajuste. Nunca aprende de validación ni test.

Las etiquetas se guardan localmente y en Drive cada ~90 s y al finalizar. La red
solo recibe el tablero y contexto anteriores; la calidad es un peso de entrenamiento.
Este proceso es ajuste supervisado, no aprendizaje por refuerzo ni autojuego.
''')
code('''
#@title 9A. Preparar aprendizaje revisado
PARTIDAS_A_REVISAR=250 #@param {type:"integer"}
PARTIDAS_VALIDACION_REVISADA=150 #@param {type:"integer"}
POSICIONES_A_REVISAR=6 #@param {type:"integer"}
SEGUNDOS_POR_BUSQUEDA=0.12 #@param {type:"number"}
if "memoria" not in globals() or "prepare_jade_quality" not in globals():
    raise RuntimeError("Ejecuta primero 1, 2, 3, 3B y 4 de Jade 2.2.")
if "tablero_jade" in globals():
    tablero_jade.settle_clock();tablero_jade.paused=True;tablero_jade.clock_started=None;tablero_jade.save_game()
revision=open_jade_quality(DB_PATH.parent,MODEL_DIR,RITMO)
try:
    prepare_jade_quality(memoria,revision,jade.binary,train_games=PARTIDAS_A_REVISAR,
        validation_games=PARTIDAS_VALIDACION_REVISADA,positions_per_game=POSICIONES_A_REVISAR,
        seconds=SEGUNDOS_POR_BUSQUEDA)
finally:
    revision.close()
print("Ahora ejecuta 9 con USAR_REVISION_CALIDAD activado.")
''')
md('''
## 9 · Aprender a predecir jugadas
Reconstruye la partida hasta ANTES del movimiento objetivo. Usa posición actual/anterior,
rating propio/rival, ritmo y relojes anteriores cuando existen. La jugada real solo se
revela como etiqueta. El error es **-log(probabilidad de la jugada humana)**.
Con USAR_REVISION_CALIDAD activo, usa solo las posiciones revisadas en 5 avanzado o 9A,
ponderadas por calidad. Con él desactivado, conserva la imitación humana original.
Las posiciones por partida de esta celda solo afectan al modo original; en el
modo revisado las posiciones ya se eligieron en 5 avanzado o 9A.
Se puntúan todas las jugadas legales. Las forzadas no inflan los porcentajes del examen.

Una ronda recorre como máximo PARTIDAS_POR_RONDA desde el cursor de entrenamiento.
Al completar el corpus vuelve al principio en la siguiente época. La selección de
posiciones por partida cambia entre épocas. Guarda pesos, optimizador y cursor.
Al interrumpir puede repetir la última partida parcialmente confirmada.

Micro-lotes y acumulación conservan el lote efectivo. GPU usa precisión mixta FP16;
si falta VRAM reduce el micro-lote sin encoger la red. Los pesos para reentrenar siguen
en FP32. El preentrenamiento opcional con frecuencias antiguas se hace una vez, hasta
el límite indicado, sin inventar historial/relojes. No se utiliza como examen.

En 2.1, **MICRO_LOTE=0** ajusta automáticamente el micro-lote manteniendo LOTE_EFECTIVO.
La preparación anticipada conserva hasta 4 bloques de 64 posiciones en RAM; no carga
todo el corpus. Puedes desactivarla para comparar el rendimiento en tu máquina.
Muestra cada ~10 segundos velocidad y tiempo restante aproximado. Al final valida una
sola vez por ejecución de la celda. Las rondas son bloques, no épocas completas.
''')
code('''
#@title 9. Entrenar o continuar
USAR_REVISION_CALIDAD=True #@param {type:"boolean"}
DISPOSITIVO_ENTRENAMIENTO="auto" #@param ["auto","cpu","cuda"]
RONDAS=2 #@param {type:"integer"}
PARTIDAS_POR_RONDA=2000 #@param {type:"integer"}
MICRO_LOTE=0 #@param [0,4,8,16,32,64,128] {type:"raw"}
LOTE_EFECTIVO=64 #@param [32,64,128] {type:"raw"}
PREPARAR_EN_PARALELO=True #@param {type:"boolean"}
POSICIONES_POR_PARTIDA=24 #@param {type:"integer"}
APROVECHAR_MEMORIA_ANTIGUA=False #@param {type:"boolean"}
MAX_POSICIONES_ANTIGUAS=20000 #@param {type:"integer"}
if "train_jade_policy" not in globals() or "memoria" not in globals():
    raise RuntimeError("Ejecuta primero 1, 2, 3, 3B y 4 del nuevo cuaderno.")
if "tablero_jade" in globals():
    tablero_jade.settle_clock();tablero_jade.paused=True;tablero_jade.clock_started=None;tablero_jade.save_game()
carpeta_entrenamiento=MODEL_DIR/"calidad_22" if USAR_REVISION_CALIDAD else MODEL_DIR
if USAR_REVISION_CALIDAD and not (carpeta_entrenamiento/"training.pt").exists() and not (MODEL_DIR/"training.pt").exists():
    raise RuntimeError("Falta training.pt original. Recupera tu carpeta de modelos o entrena primero sin revisión.")
revision=open_jade_quality(DB_PATH.parent,MODEL_DIR,RITMO) if USAR_REVISION_CALIDAD else None
try:
    resultado_entrenamiento=train_jade_policy(memoria,carpeta_entrenamiento,epochs=RONDAS,max_games=PARTIDAS_POR_RONDA,
        batch_size=MICRO_LOTE,effective_batch=LOTE_EFECTIVO,positions_per_game=POSICIONES_POR_PARTIDA,
        device=DISPOSITIVO_ENTRENAMIENTO,legacy_warmup=APROVECHAR_MEMORIA_ANTIGUA,max_legacy=MAX_POSICIONES_ANTIGUAS,
        prefetch=PREPARAR_EN_PARALELO,quality=revision,
        initial_checkpoint=MODEL_DIR/"training.pt" if USAR_REVISION_CALIDAD else None)
finally:
    if revision is not None:
        revision.close()
jade.policy=load_jade_play_policy(MODEL_DIR,RITMO)
informe_validacion=mostrar_metricas_jade(resultado_entrenamiento["metadata"]["validation"],
    folder=carpeta_entrenamiento/"informes",title="Validación ponderada por calidad" if USAR_REVISION_CALIDAD else "Validación de imitación humana")
print("El informe describe el candidato; el mensaje de activación indica si sustituyó al modelo anterior.")
print("Vuelve a ejecutar 6 para jugar o reanudar.")
''')
md('''
## 10 · Examen independiente
Validación ajusta la mezcla red/frecuencias. Solo se activa con al menos 200 ejemplos,
mejora frente a las frecuencias y al menos 30 ejemplos con mejora por perfil. Un modelo
peor que el activo en esa validación no lo sustituye. No es una medición de ELO.

INT8 se selecciona solo si coincide al menos en el 99 % de primeras opciones FP32
y aumenta el error medio como máximo 0,005 en validación. Si falla se conserva FP32.
Es una tolerancia observada, no una garantía universal. INT8 ahorra pesos pero puede
ser más lento en algunos procesadores. Se conservan ambas versiones.

El test siguiente no actualiza ni elige modelos. Úsalo al terminar los ajustes, no
como guía para repetir cambios sobre el mismo examen. Mide top 1, top 3 y error por
perfil, fase y posiciones conocidas/nuevas. No sustituye torneos ni análisis de ELO.
''')
code('''
#@title 10. Examinar sin aprender de test
PARTIDAS_EXAMEN=500 #@param {type:"integer"}
MOSTRAR_JSON=False #@param {type:"boolean"}
modelo_examen=load_jade_play_policy(MODEL_DIR,RITMO)
if modelo_examen is None:
    print("Todavía no hay predictor aprobado. Añade datos en 5 y entrena en 9.")
else:
    archivo_anterior=MODEL_DIR/"ultimo_test.json"
    anterior=json.loads(archivo_anterior.read_text()) if archivo_anterior.exists() else None
    examen=evaluate_jade_policy(modelo_examen,memoria,split="test",max_games=PARTIDAS_EXAMEN,log=print)
    informe_test=mostrar_metricas_jade(examen,folder=MODEL_DIR/"informes",previous=anterior,
                                     title="Examen independiente de Jade")
    jade_atomic_json(MODEL_DIR/"ultimo_test.json",examen)
    if MOSTRAR_JSON:
        print(json.dumps(examen,ensure_ascii=False,indent=2))
''')
md('''
## 11 · Exportar para jugar con pocos recursos
El paquete contiene pesos, metadatos, frecuencias y runtime ligero NumPy. No necesita
PyTorch ni VRAM al jugar. No es una app Android: necesita Python, numpy, python-chess
y un binario Stockfish compatible instalado aparte. El juego local es por consola UCI.
Un procesador más lento puede alcanzar menos profundidad con el mismo tiempo de búsqueda.
Esta copia es solo para jugar, no para continuar el entrenamiento o la ingesta.
''')
code('''
#@title 11. Exportar paquete CPU
import zipfile,tempfile
carpeta_exportar=MODEL_DIR/"calidad_22" if (MODEL_DIR/"calidad_22"/"active.json").exists() else MODEL_DIR
activo=carpeta_exportar/"active.json"
if not activo.exists():
    raise RuntimeError("Todavía no hay un modelo aprobado. Primero 5 y 9.")
info=json.loads(activo.read_text());directorio=carpeta_exportar/info["directory"]
destino=BASE_DIR/f"Jade_CPU_{RITMO}.zip"
with tempfile.TemporaryDirectory() as temporal:
    copia=Path(temporal)/f"jade2_{RITMO}.sqlite"
    conexion=sqlite3.connect(str(copia))
    try:
        memoria.conn.backup(conexion)
        conexion.execute("DELETE FROM corpus")
        conexion.execute("DELETE FROM quality_labels")
        conexion.execute("INSERT OR REPLACE INTO meta VALUES ('inference_only','true')")
        conexion.commit();conexion.execute("VACUUM")
    finally:
        conexion.close()
    with zipfile.ZipFile(destino,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=6) as paquete:
        paquete.write(copia,copia.name)
        for nombre in ("policy_fp32.npz","policy_int8.npz","metadata.json"):
            paquete.write(directorio/nombre,nombre)
        paquete.writestr("Jade_Ligero.py",JADE_LIGHT_SOURCE)
        paquete.writestr("LEEME.txt","Instala numpy, python-chess y Stockfish para tu CPU. Ejecuta python Jade_Ligero.py --help. Copia solo para jugar, no para entrenar ni incorporar partidas.")
print("Paquete:",destino)
from google.colab import files
files.download(str(destino))
''')
md('''
## Conservación y límites
- Copias antiguas: Jade/copias/*.sqlite.gz, intactas.
- Copias nuevas: Jade/copias_v2/*.sqlite.zst, tres generaciones por ritmo.
- Modelos: Jade/modelos/rapid o blitz. training.pt continúa el entrenamiento,
  modelo_* conserva candidatos y active.json señala el aprobado.
- Partidas y análisis: las mismas carpetas Jade/partidas y Jade/analisis.
- Revisión táctica: modelos/ritmo/revision_calidad (copias de etiquetas).
- Red corregida: modelos/ritmo/calidad_22. El training.pt original permanece intacto.
  Para volver al modelo humano original en una sesión: jade.policy=load_jade_policy(MODEL_DIR,RITMO).

La celda 10 conserva el test humano SIN filtrar. Un menor acierto de imitación
puede acompañar una mejor selección táctica. Para valorar cómo juega, analiza
partidas completas en 8; NLL y Top-1 no miden el ELO ni los errores del rival híbrido.

Los candidatos se conservan. El corpus se lee por filas, sin cargarlo entero en RAM.
Las copias son completas, no incrementales. Al reiniciar ejecuta 1, 2, 3, 3B y 4.
Cambiar a GPU reinicia Colab; guarda antes. No necesitas repetir la ingesta.

No se aplica poda no estructurada: poner ceros no garantiza menor RAM/latencia en NumPy.
No se destila Stockfish, que cambiaría el objetivo humano. Se aplica diseño pequeño,
salida sobre jugadas legales y cuantización validada. No se promete calidad idéntica
sin medirla, ni rendimiento móvil nativo.

### Referencias
- https://docs.pytorch.org/docs/stable/notes/amp_examples.html
- https://arxiv.org/abs/2006.01855
- https://www.sqlite.org/lang_vacuum.html
- https://facebook.github.io/zstd/
- https://python-chess.readthedocs.io/en/latest/engine.html
- https://huggingface.co/datasets/Lichess/standard-chess-games

Pruebas locales: migración, entrenamiento CPU, exportación e integración Stockfish.
La ruta CUDA/FP16 y tu conexión Colab/Drive requieren verificación en ese entorno.
''')
for i,cell in enumerate(cells):
    cell['id']=f'jade2-{i:02d}'
base['cells']=cells
(REPO_ROOT/'Jade_Colab.ipynb').write_text(json.dumps(base,ensure_ascii=False,indent=2))
print('Jade 2.2.1 construido:',len(cells),'celdas')
