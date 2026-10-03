# Jade 2.2.1: enseñanza avanzada opcional

**Español** | [English](Jade_2_2_LEEME.en.md) · [README principal](../README.md)

Esta actualización conserva la memoria v2, las partidas y los modelos 2.0/2.1.
No contiene tus datos ni tus pesos: los recupera de tu carpeta existente de Drive.
No vuelve a convertir una memoria v2 ni necesita descargar nuevamente las partidas.

## Nuevo: revisar mientras incorporas partidas, en la celda 5

La casilla **MODO_ENSENANZA_AVANZADA** hace que Stockfish revise cada partida nueva
de entrenamiento o validación antes de incorporarla. Abarca las primeras 100 medias
jugadas guardadas, solo los turnos de los perfiles admitidos (1200–1600). Las jugadas
forzadas reciben peso 1 sin una búsqueda. No revisa el test para aprender de él.

Stockfish funciona sin límite de ELO, con presupuesto breve. Reducirle el ELO hace
que elija jugadas inferiores deliberadamente y no lo convierte en un buen juez de
calidad humana. Jade conserva los ratings propios y del rival como contexto de la
red, además de sus perfiles al jugar. Esta ponderación no calibra un ELO humano.

Para actualizar desde 2.2:

1. Guarda memoria con 7 y espera a que termine cualquier entrenamiento en el cuaderno antiguo.
2. Abre el nuevo cuaderno y ejecuta **1 → 2 → 3 → 3B → 4**, con la misma carpeta y ritmo.
   3B recupera la copia v2 si hace falta; no vuelve a convertirla.
3. En **5** activa el modo avanzado y empieza con estos valores:

| Campo | Valor inicial |
|---|---:|
| MODO_ENSENANZA_AVANZADA | Activado |
| PARTIDAS_AVANZADAS | 100 |
| SEGUNDOS_ANALISIS_AVANZADO | 0,08 |
| ERRORES_GRAVES_CONSERVADOS_PCT | 2 |

PARTIDAS_AVANZADAS limita el lote aceptado, incluidos los grupos reservados;
también se respeta NUEVAS_PARTIDAS de la celda 2 si es menor. No toma 100 partidas
de cada perfil. Las partidas descartadas por ritmo, rating o duplicado no se analizan.

4. Ejecuta **9** con **USAR_REVISION_CALIDAD=True**, RONDAS=2, MICRO_LOTE=0 y
   LOTE_EFECTIVO=64. Continúa el modelo corregido si existe; si no, parte de tus pesos
   originales. La celda 5 actualiza frecuencias y prepara etiquetas, no entrena la red.
5. Si faltan ejemplos de validación, añade otro lote o utiliza 9A con 150 partidas
   de validación y 6 posiciones por partida. El modelo solo se activa si supera sus
   controles de validación; 100 partidas iniciales no garantizan que se apruebe.

### Qué peso tiene cada movimiento

| Evaluación rápida frente a la mejor opción encontrada | Peso |
|---|---:|
| Pérdida de 0–40 centipeones | 1 |
| Más de 40 y hasta 80 cp | 0,6 |
| Más de 80 y hasta 140 cp | 0,25 |
| Más de 140 y hasta 250 cp | 0,05 |
| Más de 250 cp | 0, salvo la muestra pequeña de entrenamiento |
| Permite mate evitable u omite un mate detectado | 0 |
| Todas las alternativas tienen mate en contra | 0,25 |

Con el ajuste 2, aproximadamente el **2 % de los errores de más de 250 cp** de
entrenamiento se conserva con **peso 0,05**. Los demás pesan cero. La muestra es
determinista: una misma jugada no vuelve a sortearse en cada época. Los errores de
mate quedan excluidos y la validación no utiliza ese sorteo. Pon 0 para excluir
todos esos errores graves. Esto no es una probabilidad de equivocarse al jugar.

Los pesos afectan a las nuevas frecuencias y al entrenamiento revisado. La jugada
real se mantiene como etiqueta: no se sustituye por una jugada de Stockfish y no es
aprendizaje por refuerzo. Las frecuencias antiguas y el corpus original permanecen;
activar la casilla no corrige retrospectivamente los dos millones de partidas.
La protección táctica de 2.2 sigue limitando las decisiones al jugar.

### Duración, interrupciones y almacenamiento

Es bastante más lento que incorporar Parquet sin análisis. Cada posición puede
necesitar una búsqueda, otra comparación y una comprobación triple si se detecta
un error grave. El presupuesto se limita por tiempo y nodos, lo primero que se
alcance. Stockfish utiliza hasta dos hilos CPU y 64 MB de hash; no utiliza la GPU.
Por ejemplo, 100 partidas con 60 posiciones revisables son 6000 posiciones. A
0,08 s por búsqueda y dos búsquedas por posición serían unos 16 minutos de cálculo,
antes de comprobaciones y otras tareas; alcanzar el límite de nodos puede reducirlo.
Es una ilustración, no una medición de tu sesión. El registro muestra el avance real.

La incorporación confirma partida, pesos y cursor en una sola transacción. Una
interrupción a mitad de partida no duplica estadísticas. El análisis parcial guarda
movimientos terminados y se respalda aproximadamente cada 90 s, y al detener con
el botón de Colab si llega KeyboardInterrupt. Las copias de memoria se hacen entre
partidas aproximadamente cada 120 s y al salir. Una caída brusca puede perder trabajo
desde el último respaldo. Al reanudar, ejecuta las celdas de inicio y de nuevo 5 con
el modo avanzado activo. No mantiene Colab conectado ni permite saltarse sus límites.

Los parámetros nuevos afectan a partidas todavía no revisadas. Una partida en
revisión se termina con sus parámetros originales para que una desconexión no
cambie los pesos. Los duplicados no se analizan otra vez.

Las etiquetas confirmadas se incluyen en la base de memoria y en la copia auxiliar
`Jade/modelos/rapid/revision_calidad` (o `blitz`). Si falta la copia auxiliar, se
reconstruye desde la memoria antes de entrenar. La caché parcial se borra al confirmar
una partida. El modelo sigue en `calidad_22`, para conservar la continuidad de 2.2.
La exportación CPU descarta corpus y etiquetas, conservando frecuencias ponderadas.

Desactivar la casilla de 5 recupera la incorporación normal. Entrenar en 9 sin
USAR_REVISION_CALIDAD vuelve a usar jugadas crudas, incluidos errores: para este
objetivo deja la revisión activada. La celda 10 conserva su test humano original.

Comprobaciones de 2.2.1: incorporación con Parquet local, pesos fraccionarios usados
por el motor, exclusión de test, deduplicación, interrupción a media partida,
restauración desde ambas copias y reconstrucción de etiquetas. También se ejecuta
entrenamiento CPU y Stockfish real para mates desde negras. No se han utilizado tus
partidas o modelos privados; sigue pendiente medir la calidad real en tus partidas.

Referencia oficial sobre fuerza limitada y CPU:
https://official-stockfish.github.io/docs/stockfish-wiki/Stockfish-FAQ.html

## Qué se ha corregido

La versión anterior entrenaba para imitar la jugada humana, buena o mala. Además,
la mezcla de frecuencias y red toleraba pérdidas altas, especialmente en aperturas.
Una frecuencia grande podía dominar la decisión, aunque su jugada fuese floja.
Más coincidencia con humanos en test no demuestra mayor fuerza del rival híbrido.

1. **Control final común.** Todas las propuestas, incluidas las de la red, pasan
   por un límite de pérdida estimada de 65–140 centipeones según el perfil.
   Dentro de ese margen también se penaliza suavemente perder ventaja. Nunca
   se fuerza un error. Estos límites son heurísticos, no una calibración de ELO.
   La probabilidad conjunta de candidatas que pierden más de 60 cp se limita al
   3–8 % según perfil; no se obliga a alcanzar ese porcentaje. El límite depende
   de las evaluaciones disponibles y no garantiza ese porcentaje en partidas reales.
2. **Referencia y comprobación táctica.** Hace una búsqueda de una sola variante,
   y si elige otra jugada la comprueba frente a la referencia con dos variantes.
   Mantiene los mates detectados y rechaza permitir un mate cuando hay alternativa.
   El cálculo puede aumentar hasta aproximadamente 1,8 veces el presupuesto previo;
   el reloj descuenta el cálculo real y la pausa. Un análisis corto sigue pudiendo fallar.
3. **Ajuste supervisado ponderado.** La nueva celda 9A revisa una muestra con
   Stockfish. La celda 9 aprende de esa muestra usando pesos de calidad. No es RL.
   Conserva la jugada humana como etiqueta, no la sustituye por la primera del motor.
4. **Visor PGN.** Barra vertical negra/blanca animada, evaluación numérica en cp
   y peones, mates separados y botones para antes/después y alternativa del motor.
   +100 cp equivale a +1,00 peones desde blancas. La altura es una escala visual
   no lineal, no porcentaje de victoria. Los HTML antiguos no se modifican solos.

## Actualizar y jugar con lo aprendido

1. Si el cuaderno anterior sigue activo, termina la tarea y guarda la memoria con 7.
   Los pesos se guardan durante el entrenamiento; 7 guarda la memoria, no entrena.
2. Descarga `Jade_Colab_2_2_1.ipynb` actualizado y ábrelo en Colab como una copia nueva.
   Conserva el cuaderno anterior como referencia y utiliza un único cuaderno activo
   por carpeta de memoria/modelos.
3. En 2 selecciona la misma cuenta, carpeta de Drive y ritmo (`rapid` o `blitz`).
4. Ejecuta **1 → 2 → 3 → 3B → 4 → 6**. La 3B recupera la copia v2; no debe repetir
   la conversión antigua. En 4 se anuncia la protección táctica de Jade 2.2.
5. Prueba el rival antes de ajustar la red: el control ya actúa sobre tus pesos actuales.

La actualización no elimina ni modifica tus partidas antiguas para que puedas
reproducir el entrenamiento original. No garantiza que no haya errores ni un ELO real.
Sin analizar tu PGN no se puede atribuir cada fallo concreto a una sola causa.

## Corregir también el aprendizaje

Ejecuta **9A** con los valores iniciales:

| Ajuste | Valor |
|---|---:|
| Partidas nuevas de entrenamiento a revisar | 250 |
| Partidas de validación revisada, objetivo total | 150 |
| Posiciones por partida | 6 |
| Segundos por búsqueda | 0,12 |

La revisión usa CPU y puede tardar varios minutos. Cada posición puede requerir
más de una búsqueda; los errores que se excluirían se comprueban con más cálculo.
No analiza automáticamente los dos millones de partidas. Repetir 9A añade otro
lote de entrenamiento y reutiliza la validación. Un juego parcialmente revisado
puede repetirse al reanudar; las partidas ya completadas no se duplican.

Luego ejecuta **9** con `USAR_REVISION_CALIDAD=True`, **RONDAS=2**,
`PARTIDAS_POR_RONDA=2000`, `MICRO_LOTE=0`, `LOTE_EFECTIVO=64` y dispositivo `auto`.
2000 es un máximo: si solo hay 250 partidas revisadas, trabaja con esas 250.
La selección de posiciones viene de 9A; `POSICIONES_POR_PARTIDA` en 9 no la amplía.
No conviene repetir cientos de rondas sobre una muestra tan pequeña: amplía 9A
y comprueba validación. La T4 no acelera el análisis de Stockfish.

| Pérdida estimada de la jugada humana | Peso para ajustar la red |
|---|---:|
| 0–40 cp | 1 |
| Más de 40, hasta 80 cp | 0,6 |
| Más de 80, hasta 140 cp | 0,25 |
| Más de 140, hasta 250 cp | 0,05 |
| Más de 250 cp, permite mate evitable u omite mate | 0 (se excluye) |

Si todas las alternativas tienen mate en contra, no se culpa a la jugada de un
resultado inevitable: recibe peso 0,25. Los umbrales y la búsqueda finita pueden
infravalorar algún sacrificio correcto. Aumentar el tiempo de 9A mejora la revisión
de las nuevas muestras, pero no vuelve a analizar automáticamente las ya guardadas.

La red corregida se guarda en `Jade/modelos/rapid/calidad_22` (o `blitz`). El
checkpoint original `Jade/modelos/rapid/training.pt` se conserva. El ajuste copia
sus pesos y optimizador cuando comienza por primera vez, usando una tasa de aprendizaje
menor. El modelo nuevo debe superar su referencia y el activo sobre validación
ponderada antes de activarse. No se seleccionan modelos por test.

Las etiquetas se respaldan en `Jade/modelos/rapid/revision_calidad`, con una copia
anterior de reserva. Se utiliza SQLite local durante la sesión; no se entrena
consultando directamente una base abierta sobre Drive.

Tras entrenar, ejecuta 6. Para volver al predictor humano original en la sesión:

```python
jade.policy = load_jade_policy(MODEL_DIR, RITMO)
```

El control táctico seguirá activo. No hace falta borrar archivos.

## Medir resultados sin engañarnos

- La validación de calidad tiene pesos distintos y no se compara numéricamente
  con el test original. El informe la identifica expresamente.
- La celda 10 mantiene el test humano sin filtrar. Puede bajar Top-1 al dejar de
  imitar ciertos fallos. No uses test para ajustar umbrales ni escoger modelos.
- El test evalúa la red, no la mezcla final con Stockfish. Para estudiar errores
  reales del rival, guarda PGN completos y analízalos con más tiempo en la celda 8.
- El perfil 1300 sigue siendo un objetivo de estilo, no un ELO certificado.

## Compatibilidad y alcance de las pruebas

La arquitectura neuronal y el esquema de memoria v2 se conservan. Se comprueban
ponderaciones, mates, normalización, separación de corpus, respaldo/restauración,
continuación CPU desde pesos anteriores y sintaxis del cuaderno. También se prueban
Stockfish 17.1 real, mates en ambos colores, revisión incremental y PGN con mate.
El visor se comprueba en Chromium a 360 y 1200 píxeles de ancho, incluida navegación
y actualización de la barra. No se dispone de
tus pesos ni de tus dos millones de partidas para medir una mejora real de fuerza.
CUDA/FP16, Colab y la sincronización con tu Drive requieren comprobarse en tu sesión.

La política de uso del proveedor sigue siendo aplicable: Colab publica restricciones
para entrenamiento de ajedrez en su modalidad gratuita sin saldo de cómputo.
Consulta https://research.google.com/colaboratory/faq.html antes de sesiones largas.

Referencias técnicas:
- https://python-chess.readthedocs.io/en/latest/engine.html
- https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
