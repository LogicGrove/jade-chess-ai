<p><img src="assets/jade-banner.png" alt="Jade · Chess + AI" width="960"></p>

# Jade Chess AI · 2.2.1

**Español** | [English](README.en.md)

**Una IA de ajedrez híbrida que busca jugar de forma humana, no solo encontrar la mejor jugada.**

Jade combina Stockfish, estadísticas de partidas reales de Lichess y una red neuronal compacta. Permite experimentar con perfiles de juego, entrenar un predictor de movimientos humanos, jugar en un tablero interactivo y analizar partidas PGN.

Proyecto educativo impulsado por **Manu**, desarrollado de forma iterativa con ayuda de herramientas de IA. Estado: **experimental**. No está afiliado a Stockfish, Lichess, Google ni Hugging Face.

## Qué incluye

- Juego híbrido: candidatos de Stockfish, muestreo de Boltzmann, frecuencias humanas y predictor aprobado.
- Perfiles 1200, 1300, 1400, 1500 y 1600 como objetivos de estilo. **No son niveles ELO medidos ni certificados.**
- Entre 3 y 15 candidatas según el tiempo restante, con una demora de respuesta simulada.
- Controles tácticos para reducir errores graves y proteger mates detectados, sin garantizar juego perfecto.
- Incorporación incremental de Parquet de `Lichess/standard-chess-games`, filtrado, deduplicación y cursores.
- Memoria SQLite compacta y copias comprimidas sin pérdida con Zstandard.
- Predicción supervisada: se reconstruye la posición antes de una jugada y se intenta predecir la jugada humana.
- Modo avanzado opcional: Stockfish revisa jugadas y asigna menos peso a errores antes de incorporarlas.
- Entrenamiento CPU o CUDA, micro-lotes, acumulación y preparación anticipada de ejemplos.
- Predictor pequeño, aproximadamente 212 000 parámetros, y exportación de pesos cuantizados INT8.
- Tablero interactivo SVG de `python-chess`, reloj, elección de color y perfil.
- Analizador PGN con tablero, alternativas, barra de evaluación blanca/negra e informes HTML, PGN y CSV.
- Exportación para jugar por consola en CPU con un Stockfish compatible.

## Empezar en Colab

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LogicGrove/jade-chess-ai/blob/main/Jade_Colab_2_2_1.ipynb)

1. Abre [Jade_Colab_2_2_1.ipynb](Jade_Colab_2_2_1.ipynb). Puedes descargarlo y subirlo a [Google Colab](https://colab.research.google.com/).
2. Ejecuta **1 → 2 → 3 → 3B → 4**. En 2 selecciona ritmo y almacenamiento.
3. Para jugar, ejecuta **6**. Jade puede funcionar sin red entrenada usando su rival híbrido de respaldo.
4. Para incorporar datos, ejecuta **5**. Empieza con lotes pequeños.
5. Para entrenar, ejecuta **9**. Para consultar el test, ejecuta **10**.
6. Guarda con **7** antes de terminar. El análisis PGN está en **8** y los informes guardados en **8B**.

El cuaderno contiene el programa completo: no necesitas subir los `.py` por separado. En una sesión nueva se deben volver a ejecutar las celdas de inicio. **3B también restaura copias v2: no convierte otra vez una memoria ya convertida.** Utiliza un único cuaderno activo por memoria.

Enlace directo al cuaderno en Colab:

```text
https://colab.research.google.com/github/LogicGrove/jade-chess-ai/blob/main/Jade_Colab_2_2_1.ipynb
```

### CPU y GPU

| Tarea | Recurso |
|---|---|
| Stockfish y revisión avanzada | CPU, no CUDA |
| Entrenamiento neuronal | CPU o GPU CUDA compatible, como T4 |
| Predictor exportado y consola | CPU, sin VRAM para el predictor |
| Almacenamiento | Disco local durante uso; copias y modelos en Drive si se activa |

No existe un interruptor que convierta Stockfish en un motor GPU. La GPU acelera la red, no el análisis de Stockfish. La precisión INT8 reduce el tamaño de los pesos, pero no garantiza aceleración INT8 nativa en todos los dispositivos.

Revisa las [condiciones actuales de Colab](https://research.google.com/colaboratory/faq.html). Su FAQ restringe expresamente el entrenamiento de ajedrez en entornos gratuitos sin saldo positivo de unidades de cómputo. La disponibilidad de GPU, límites y duración de sesiones no están garantizados. Este proyecto no incluye mecanismos para eludirlos.

## Datos y aprendizaje

El código lee lotes remotos de [Lichess/standard-chess-games](https://huggingface.co/datasets/Lichess/standard-chess-games). No necesitas guardar todos los Parquet en Drive. Año y mes seleccionan la fuente; rapid y blitz mantienen memorias/modelos separados.

Se almacenan las primeras 100 medias jugadas por partida, ratings y relojes cuando existen. Las secuencias se separan aproximadamente **80 % entrenamiento, 10 % validación y 10 % test**, mediante una huella del contenido. Secuencias idénticas no cruzan grupos. Jugadores y aperturas comunes sí pueden aparecer en varios grupos: no es una separación por jugador.

Validación y test no alimentan las frecuencias ni el entrenamiento neuronal. Las estadísticas antiguas migradas no se utilizan como un examen independiente. Los ratings de Lichess son Glicko-2 y no equivalen automáticamente al ELO FIDE.

**El repositorio no incluye partidas privadas, bases de memoria ni pesos preentrenados.** Incorporar datos en 5 no equivale a entrenar la red: se entrena en 9. Jade no aprende automáticamente de tus partidas contra ella.

### Enseñanza avanzada

En 5, `MODO_ENSENANZA_AVANZADA` revisa las jugadas retenidas de perfiles admitidos antes de incorporarlas. Stockfish trabaja sin límite artificial de ELO, con una búsqueda breve: debilitar al juez no mejora sus etiquetas.

| Pérdida estimada respecto a la mejor opción encontrada | Peso |
|---|---:|
| 0–40 centipeones | 1 |
| >40–80 | 0,6 |
| >80–140 | 0,25 |
| >140–250 | 0,05 |
| >250 | 0, salvo la muestra opcional de entrenamiento |

Con el ajuste por defecto, aproximadamente el 2 % de los errores graves de entrenamiento se conserva con peso 0,05. **No significa que Jade cometa errores el 2 % de las veces.** Los errores de mate detectados se excluyen; la validación no utiliza ese sorteo. Las búsquedas rápidas pueden equivocarse.

El objetivo sigue siendo imitar jugadas humanas ponderadas por calidad: **no es aprendizaje por refuerzo ni sustitución automática de etiquetas por jugadas de Stockfish**. No reanaliza retrospectivamente toda la memoria antigua.

En una instalación nueva, entrena primero el predictor original en 9 con `USAR_REVISION_CALIDAD=False`. Después puedes preparar revisiones y entrenar con `USAR_REVISION_CALIDAD=True`; el ajuste de calidad necesita ese checkpoint de partida. Empieza con 100 partidas avanzadas y consulta [la guía detallada](Jade_2_2_LEEME.md).

### Entender las métricas

| Métrica | Dirección deseable | Qué mide |
|---|---|---|
| NLL / error de predicción | Bajar | Probabilidad asignada a las jugadas humanas observadas |
| Top-1 | Subir | La jugada humana es la propuesta más probable |
| Top-3 | Subir | La jugada humana está entre las tres más probables |
| `n` | Contexto, no calidad por sí solo | Número de posiciones evaluadas |

Una mejor predicción humana no demuestra por sí sola mayor fuerza ajedrecística. Compara métricas con el mismo conjunto y objetivo; no compares directamente validación ponderada y test humano sin ponderar. Usa validación para elegir modelos y reserva test para evaluaciones finales, sin ajustar parámetros repetidamente a él.

## Guardado y privacidad

Por defecto, con Drive activo:

- Carpeta persistente: `/content/drive/MyDrive/Jade`.
- Modelos: `modelos/rapid` o `modelos/blitz`; ajuste de calidad en `calidad_22`.
- Copias: `copias_v2/*.sqlite.zst`, conservando tres por ritmo.
- Informes: `analisis`; partidas de usuario: `partidas`.
- Base de trabajo: `/content/Jade`, temporal y descomprimida.

GitHub guarda **código**, no es la copia de seguridad de tu corpus. No publiques Drive completo, claves, tokens, partidas personales o checkpoints. Aunque no se guardan nombres de jugadores en el corpus, los IDs pueden permitir enlazar a partidas públicas: no se debe describir como completamente anónimo.

## Código del proyecto

| Archivo | Función |
|---|---|
| `Jade_Colab_2_2_1.ipynb` | Cuaderno autónomo para usuarios |
| `Jade_Programa.py` | Programa completo generado |
| `jade_core.py`, `jade_storage_v3.py` | Base, datos, migración y copias |
| `jade_policy.py`, `jade_reports.py` | Red, entrenamiento y métricas |
| `jade_quality.py`, `jade_advanced.py` | Revisión y ponderación por calidad |
| `jade_engine_v2.py`, `jade_hybrid_v3.py` | Motor híbrido y controles tácticos |
| `jade_ui.py`, `jade_pgn.py` | Interfaz y analizador PGN |
| `Jade_Ligero.py` | Versión de consola para CPU |
| `build_jade_v3.py`, `Jade_v12_base.ipynb` | Constructor y plantilla histórica; sus nombres no indican la versión publicada |
| `verify_*.py` | Comprobaciones focalizadas de desarrollo |

Consulta [CONTRIBUTING.md](CONTRIBUTING.md) antes de editar fuentes o regenerar el cuaderno. `requirements.txt` documenta las dependencias del cuaderno, no instala Stockfish ni reinstala PyTorch.

## Consola CPU

Exporta primero un modelo aprobado y la memoria con la celda 11. Instala un Stockfish compatible con tu sistema y las dependencias ligeras:

```bash
python -m pip install -r requirements-cpu.txt
python Jade_Ligero.py --stockfish /ruta/a/stockfish --memory /ruta/jade2_rapid.sqlite --model-dir /ruta/modelo --profile 1300
```

La carpeta del modelo necesita `metadata.json` y los pesos exportados correspondientes. Añade `--black` para jugar con negras. No es una aplicación Android ni se promete el mismo rendimiento en todos los dispositivos.

## Licencia y créditos

Copyright (C) 2026 Manu y colaboradores de Jade, para sus contribuciones al proyecto.

El código propio y la documentación de Jade se distribuyen bajo **GNU GPL versión 3 o, a tu elección, cualquier versión posterior** (`GPL-3.0-or-later`). Consulta [LICENSE](LICENSE). Se distribuye sin garantía. Las dependencias, Stockfish y los datos conservan sus propias licencias, descritas en [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

Agradecimientos a los desarrolladores de Stockfish, python-chess, Lichess, PyTorch y el resto de herramientas utilizadas. No se incluyen binarios de terceros ni sus bases de datos en esta publicación.

## Limitaciones y próximos pasos

Jade es un prototipo educativo, no un motor de competición ni una réplica exacta de un jugador humano. Pendientes: medir fuerza con partidas controladas, calibrar perfiles, evaluar generalización fuera del corpus y comparar la cuantización en cada entorno objetivo. No se incluyen resultados reproducibles de tus modelos privados ni garantías de tiempo de reacción.

Para informar de un fallo, abre un Issue con versión, celda, entorno y traceback, sin incluir datos personales. Más instrucciones en [CONTRIBUTING.md](CONTRIBUTING.md).
