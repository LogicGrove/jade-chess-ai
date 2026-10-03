# Publicar Jade 2.2.1 en GitHub

Este paquete contiene el código actual y documentación lista para revisar. No se ha publicado nada automáticamente en GitHub. Revisa y acepta la licencia propuesta y el nombre de crédito antes de hacer público el repositorio.

## 1. Datos del repositorio

| Campo | Recomendación |
|---|---|
| Nombre | `jade-chess-ai` |
| Nombre visible en README | `Jade Chess AI` |
| Versión inicial del repositorio | `v2.2.1` |
| Estado | Experimental / proyecto educativo |
| Visibilidad | Pública si quieres compartir el código; privada durante la revisión inicial |
| Rama | `main` |
| Licencia del código propio | `GPL-3.0-or-later` |

Descripción breve en español:

> IA de ajedrez híbrida con Stockfish, estadísticas de Lichess y una red neuronal compacta. Perfiles humanos experimentales, entrenamiento en Colab y análisis visual de PGN.

Descripción breve en inglés:

> Experimental human-like chess AI combining Stockfish, Lichess game statistics and a compact neural policy, with Colab training and visual PGN analysis.

Topics recomendados:

```text
chess chess-ai stockfish python machine-learning pytorch google-colab lichess human-like-ai parquet sqlite educational
```

## 2. Licencia recomendada

Recomiendo **GNU GPL v3 o posterior**. El archivo `LICENSE` contiene el texto íntegro de GPL v3, y README declara la opción «o posterior» con identificador `GPL-3.0-or-later`.

Es una elección coherente con python-chess, que declara GPL v3 o posterior, y con la intención de compartir Jade y mantener abiertas las versiones derivadas cubiertas por GPL cuando se distribuyan. Stockfish conserva su propia GPL y se instala aparte; no se deduce que cualquier programa que invoque Stockfish por UCI sea necesariamente GPL.

En términos sencillos:

- Otros pueden usar, estudiar, modificar y redistribuir el código, incluso comercialmente.
- Deben conservar avisos y cumplir las obligaciones de licencia y acceso al código fuente correspondiente cuando distribuyan versiones cubiertas.
- No obliga a publicar todas las modificaciones privadas ni a usar GitHub específicamente.
- No impide que alguien venda una versión y no es una licencia «solo para uso no comercial».
- No concede automáticamente derechos sobre marcas, datos privados o servicios externos.
- No ofrece garantía de funcionamiento ni de fuerza ajedrecística.

No recomendaría MIT como licencia general de esta distribución integrada sin revisar antes la compatibilidad y separación del código GPL. Los datos Lichess citados declaran CC0 y los componentes externos conservan sus propias licencias. Si en el futuro distribuyes un binario de Stockfish, añade su licencia y proporciona su fuente correspondiente conforme a GPL. No está incluido en este ZIP.

El crédito propuesto es «Copyright (C) 2026 Manu y colaboradores de Jade, para sus contribuciones». Ajusta alias y titulares legítimos si corresponde. Este texto no adjudica a Manu la autoría de Stockfish, librerías o partidas ni resuelve titularidad legal por sí mismo. Para dudas de titularidad o una publicación comercial con requisitos específicos, consulta asesoramiento jurídico.

## 3. Qué subir y qué conservar fuera

Sube el contenido de la carpeta del paquete: notebook, fuentes Python, builder, scripts de comprobación, documentación, LICENSE, requirements y `.gitignore`.

**No subas el ZIP como único archivo del repositorio.** Descomprímelo y sube los archivos que contiene. Los usuarios necesitan poder leer el código y abrir el notebook.

Mantén fuera:

- La carpeta privada `MyDrive/Jade`, bases SQLite y copias comprimidas.
- Tus millones de partidas, Parquet, corpus y partidas de usuarios.
- `training.pt`, pesos `.npz`, archivos de optimizador y modelos privados.
- Informes de análisis y datos personales.
- Tokens, contraseñas, credenciales, cookies y enlaces privados.
- Binarios de Stockfish u otras librerías, salvo que prepares una distribución conforme a sus licencias.

El `.gitignore` ayuda al trabajar con Git local, pero **no protege los archivos que subas manualmente desde la web ni elimina archivos ya registrados**. Revisa siempre lo que vas a enviar.

GitHub no es un disco de copias de seguridad. Su documentación limita la subida web a 25 MiB por archivo, advierte por encima de 50 MiB en Git y bloquea archivos de más de 100 MiB en repositorios normales. LFS o Releases pueden servir para algunos archivos grandes, con sus condiciones y cuotas; no es necesario para este paquete de código.

## 4. Publicar desde la web, sin terminal

1. Un titular de cuenta que cumpla los requisitos entra en GitHub y pulsa **New repository**.
2. Escribe `jade-chess-ai` y pega la descripción breve. Elige visibilidad.
3. Crea un repositorio vacío: no marques README, licencia ni `.gitignore` automáticos porque el paquete ya los incluye.
4. Descomprime el paquete. En GitHub usa **uploading an existing file** o **Add file → Upload files** y selecciona el contenido de la carpeta, no una carpeta extra que anide todo.
5. Comprueba que README y notebook quedan en la raíz. Algunos exploradores ocultan `.gitignore`: asegúrate de añadirlo; puedes crearlo desde **Add file → Create new file** si no aparece.
6. Mensaje de commit sugerido: `Publicación inicial de Jade 2.2.1`.
7. En **About**, añade descripción y topics. No declares ELO certificado, modelo preentrenado incluido ni benchmarks que no puedas reproducir.
8. Prueba el notebook con el enlace Colab construido abajo. Guardar cambios en Colab no actualiza automáticamente GitHub.
9. Opcional: crea una Release con tag `v2.2.1`, título `Jade 2.2.1: enseñanza avanzada` y notas basadas en CHANGELOG. Etiqueta esta misma revisión publicada.

En un móvil es posible que seleccionar muchos archivos sea incómodo. Puedes empezar con el notebook y documentación, y después añadir fuentes; un ordenador facilita revisar toda la publicación. No borres ni muevas tu memoria de Drive para publicar el código.

## 5. Alternativa con Git local

Ejecuta desde la carpeta descomprimida que contiene README. Sustituye `TU_USUARIO` por el propietario real y autentícate con un método admitido por GitHub; no escribas tokens en los comandos ni los publiques.

```bash
git init
git branch -M main
git status --short
git add .
git diff --cached --stat
git diff --cached --name-only
git commit -m "Publicación inicial de Jade 2.2.1"
git remote add origin https://github.com/TU_USUARIO/jade-chess-ai.git
git push -u origin main
```

No ejecutes el push si ves datos privados en la revisión. Un archivo ya publicado puede permanecer en el historial aunque se elimine después; si se filtra una credencial, revócala inmediatamente.

## 6. Enlace para abrir Colab

Sustituye propietario y rama si procede:

```text
https://colab.research.google.com/github/TU_USUARIO/jade-chess-ai/blob/main/Jade_Colab_2_2_1.ipynb
```

Cuando el enlace esté probado, puedes añadir este botón al README, sustituyendo el marcador:

```markdown
[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TU_USUARIO/jade-chess-ai/blob/main/Jade_Colab_2_2_1.ipynb)
```

Colab es el entorno de ejecución; GitHub aloja código e historial; Drive conserva tus datos y modelos cuando lo activas. Publicar GitHub no convierte automáticamente un modelo privado en descargable ni mantiene Colab conectado.

## 7. Presentación del proyecto y participación de Manu

Texto opcional para la presentación, si quieres hacer pública la edad:

> Jade es un proyecto educativo de ajedrez e inteligencia artificial iniciado por Manu a los 12 años. Explora cómo combinar un motor fuerte con estadísticas reales y aprendizaje automático para crear un rival de estilo más humano. Se ha desarrollado de forma iterativa con ayuda de herramientas de IA, incorporando gestión de datos, entrenamiento, evaluación e interfaces de juego.

El interés está en el planteamiento, las pruebas y la evolución del sistema. No hace falta afirmar que todo el código se escribió sin ayuda ni que se inventó Stockfish. Evita apellido, centro educativo, ubicación, edad exacta o información de contacto si no quieres exponerlos.

GitHub exige al menos **13 años**, y el mínimo puede ser superior en algunos países. Si Manu todavía tiene 12, un adulto puede publicar y administrar el repositorio desde **su propia cuenta**, acreditando a Manu. No crees una cuenta con fecha falsa ni compartas una cuenta: las condiciones también exigen un único usuario por inicio de sesión.

## 8. Revisión final

- [ ] He leído y aceptado GPL-3.0-or-later para el código propio y revisado el crédito.
- [ ] No hay datos, modelos privados, secretos, salidas de notebook ni binarios externos.
- [ ] README, notebook 2.2.1, fuentes y licencia están en la raíz.
- [ ] Hay una explicación de instalación, aprendizaje, guardado y limitaciones.
- [ ] El enlace Colab utiliza el usuario y la rama reales.
- [ ] Los perfiles son objetivos de estilo, no ELO certificado.
- [ ] No anuncio pesos preentrenados que el repositorio no contiene.
- [ ] Si publico resultados, incluyo versión, modelo, datos reservados, presupuesto de análisis y condiciones de la medición.
- [ ] La cuenta publicadora cumple las condiciones de GitHub.

## Fuentes oficiales

Consultadas al preparar esta guía; condiciones y límites pueden cambiar:

- [Crear un repositorio](https://docs.github.com/en/repositories/creating-and-managing-repositories/creating-a-new-repository).
- [Archivos grandes en GitHub](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github).
- [Condiciones de GitHub](https://docs.github.com/en/site-policy/github-terms/github-terms-of-service).
- [GPL v3 en Choose a License](https://choosealicense.com/licenses/gpl-3.0/).
- [Licencia de python-chess](https://python-chess.readthedocs.io/en/stable/).
- [Stockfish y distribución](https://stockfishchess.org/about/).
- [Dataset Lichess en Hugging Face](https://huggingface.co/datasets/Lichess/standard-chess-games).
- [Exports de Lichess](https://database.lichess.org/).
- [FAQ de Colab](https://research.google.com/colaboratory/faq.html).
