# Historial resumido

## 2.2.1

- Enseñanza avanzada opcional en la celda 5.
- Revisión rápida con Stockfish antes de incorporar jugadas de perfiles admitidos.
- Frecuencias y etiquetas ponderadas según pérdida estimada de centipeones.
- Muestra pequeña y determinista de errores graves de entrenamiento.
- Reanudación del análisis parcial y confirmación conjunta de partidas, pesos y cursor.
- El test no se incorpora al aprendizaje revisado.

## 2.2

- Protección táctica común para propuestas de frecuencias y red.
- Ajuste supervisado ponderado por calidad manteniendo el checkpoint original.
- Visor PGN con barra de evaluación, tablero y navegación por alternativas.

## Origen de la serie 2.x

- Memoria compacta, migración sin borrar el original y copias Zstandard.
- Predictor humano entrenable, validación/test reservados y exportación CPU.
- Interfaz interactiva, tiempo de reacción y candidatas adaptadas al reloj.

## Preparación para GitHub

Se añaden README, licencia propuesta, avisos, dependencias, guía de contribución y exclusiones. El código funcional y el cuaderno 2.2.1 se conservan sin cambios. No se incluyen datos ni modelos privados, y no se declara un ELO certificado.
