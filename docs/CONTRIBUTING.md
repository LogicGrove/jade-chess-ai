# Contribuir a Jade

Gracias por ayudar a mejorar este prototipo educativo. Usa Issues para fallos o propuestas y Pull Requests para cambios. No publiques bases privadas, tokens ni partidas personales.

## Flujo de desarrollo

1. Trabaja en una rama y conserva las copias originales de memoria y modelos.
2. Instala las dependencias de `requirements-dev.txt`. Para pruebas de entrenamiento, instala PyTorch según [sus instrucciones oficiales](https://pytorch.org/get-started/locally/). Instala Stockfish compatible aparte.
3. Edita las fuentes `jade_*.py` pertinentes. El constructor selecciona y concatena partes de estas fuentes: no todos los módulos son bibliotecas autónomas importables por separado.
4. Ejecuta `python build_jade_v3.py` desde la raíz del repositorio. Genera `Jade_Programa.py`, `Jade_Ligero.py` y `Jade_Colab.ipynb`.
5. Si estás actualizando la versión publicada, revisa el notebook generado y copia su contenido al nombre versionado correspondiente. El builder no actualiza automáticamente `Jade_Colab_2_2_1.ipynb`.
6. Revisa diferencias, limpia salidas de notebooks y no agregues checkpoints o informes privados.

No edites solo el programa concatenado: la siguiente regeneración podría perder ese cambio. Los nombres históricos `v3`, `v2` y `v12` no indican la versión comercial/publicada del proyecto.

## Comprobaciones existentes

```bash
python verify_jade_advanced.py
python verify_jade_v22.py
```

Estos scripts usan datos de prueba temporales, algunas búsquedas reales con Stockfish y entrenamiento CPU. Revisa sus rutas y dependencias antes de ejecutarlos; requieren más que las dependencias ligeras.

`verify_jade_v22.py` genera `qa_report.html`. La prueba visual `verify_pgn_v22.py` lo necesita: ejecuta primero el anterior. También necesita Playwright y Chromium instalados. Los scripts no sustituyen una prueba completa dentro de Colab o una medición de fuerza.

Esta publicación incluye los scripts, pero no promete que todas las plataformas y versiones futuras de dependencias hayan pasado las pruebas. Describe siempre qué ejecutaste y en qué entorno.

## Qué informar

- Versión de Jade y celda/script afectado.
- Versión de Python, CPU/GPU y Stockfish.
- Pasos mínimos para reproducir y traceback completo, sin secretos.
- Resultado esperado y observado.

Para cambios de aprendizaje, separa métricas de imitación, calidad táctica y fuerza de juego. No ajustes modelos con test. Documenta los presupuestos de análisis, semilla, datos y diferencias de cuantización.

Las contribuciones al código propio se proponen bajo GPL-3.0-or-later. No incorpores código de terceros sin licencia compatible y avisos correspondientes.
