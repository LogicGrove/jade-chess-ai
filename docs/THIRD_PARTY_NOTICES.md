# Componentes de terceros y datos

La licencia de Jade no cambia las licencias de los proyectos externos. No se incluyen sus binarios ni su código fuente completo en este repositorio. Consulta la licencia de la versión concreta instalada y las dependencias transitivas si preparas una distribución empaquetada.

## Componentes principales

| Componente | Función | Licencia / fuente oficial |
|---|---|---|
| python-chess | Reglas, PGN, SVG y comunicación UCI | GPL-3.0-or-later; [documentación](https://python-chess.readthedocs.io/en/stable/) y [licencia](https://github.com/niklasf/python-chess/blob/master/LICENSE.txt) |
| Stockfish | Motor externo CPU, búsquedas y revisión | GPL v3; [condiciones de distribución](https://stockfishchess.org/about/) |
| NumPy | Inferencia y operaciones numéricas | BSD-3-Clause; [licencia](https://github.com/numpy/numpy/blob/main/LICENSE.txt) |
| PyTorch | Entrenamiento neuronal opcional | [Repositorio y licencia](https://github.com/pytorch/pytorch) |
| Apache Arrow / PyArrow | Lectura Parquet | [Repositorio y licencia](https://github.com/apache/arrow) |
| huggingface_hub | Acceso al dataset | Apache-2.0; [repositorio](https://github.com/huggingface/huggingface_hub) |
| ipywidgets | Controles interactivos | [Repositorio y licencia](https://github.com/jupyter-widgets/ipywidgets) |
| fsspec | Acceso a archivos remotos | [Repositorio y licencia](https://github.com/fsspec/filesystem_spec) |
| python-zstandard | Compresión de copias | [Repositorio y licencia](https://github.com/indygreg/python-zstandard); también usa Zstandard |
| threadpoolctl | Control de hilos de librerías numéricas | [Repositorio y licencia](https://github.com/joblib/threadpoolctl) |
| nbformat | Comprobaciones de notebooks | [Repositorio y licencia](https://github.com/jupyter/nbformat) |
| Playwright | Comprobación visual opcional de desarrollo | [Repositorio](https://github.com/microsoft/playwright-python) |

## Stockfish se instala aparte

Jade se comunica con un proceso externo mediante UCI. Este paquete no contiene un ejecutable de Stockfish ni modifica su código. Usar un motor externo no significa por sí solo que todo programa que lo invoque tenga que adoptar su licencia. La recomendación GPL para Jade también tiene en cuenta su uso directo de python-chess.

Si distribuyes un ejecutable de Stockfish, debes cumplir su GPL: conservar avisos/licencia y proporcionar acceso al código fuente correspondiente a esa compilación conforme a sus condiciones. Un enlace genérico a la página del proyecto no sustituye automáticamente esa obligación.

## Datos de Lichess

Fuente: [Lichess/standard-chess-games](https://huggingface.co/datasets/Lichess/standard-chess-games), que declara **CC0-1.0**. Los [exports oficiales de Lichess](https://database.lichess.org/) también declaran CC0. No se incluyen Parquet ni partidas de usuario en este paquete.

CC0 es independiente de GPL, y permite reutilización de los datos. Se conserva el crédito por transparencia. No supongas que cualquier otro dataset o exportación personal tiene las mismas condiciones. Revisa privacidad antes de publicar datos, IDs y resultados.

## Modelos futuros

No se distribuyen pesos entrenados aquí. Si se publican en el futuro, se deben añadir una ficha del modelo, procedencia de datos, métricas reproducibles, limitaciones y una declaración explícita de licencia. No se debe asumir que los pesos heredan automáticamente CC0 por entrenarse con datos CC0 ni que el texto del LICENSE del código resuelve cualquier distribución de modelos.

## Servicios

Google Colab, Google Drive, Hugging Face y GitHub son servicios externos con sus propias condiciones. Esta documentación no concede sus marcas, recursos de cómputo ni almacenamiento.
