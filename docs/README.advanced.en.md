<p><img src="../assets/jade-banner.png" alt="Jade · Chess + AI" width="960"></p>

# Jade Chess AI · 2.2.1 — Advanced guide

[Español](README.advanced.md) | **English** · [Quick start](../README.en.md)

**A hybrid chess AI that aims to play like a human, rather than simply find the best move.**

Jade combines Stockfish, statistics from real Lichess games and a compact neural network. It lets you experiment with playing profiles, train a human move predictor, play on an interactive board and analyze PGN games.

An educational project led by **Manu**, developed iteratively with the help of AI tools. Status: **experimental**. It is not affiliated with Stockfish, Lichess, Google or Hugging Face.

## Features

- Hybrid play: Stockfish candidates, Boltzmann sampling, human move frequencies and an approved predictor.
- Profiles 1200, 1300, 1400, 1500 and 1600 as style targets. **These are not measured or certified Elo ratings.**
- Between 3 and 15 candidate moves depending on the time remaining, with a simulated response delay.
- Tactical checks to reduce serious mistakes and preserve detected mates, without guaranteeing perfect play.
- Incremental ingestion of Parquet files from `Lichess/standard-chess-games`, with filtering, deduplication and progress cursors.
- Compact SQLite memory and lossless compressed backups using Zstandard.
- Supervised prediction: reconstruct the position before a move and try to predict the human move.
- Optional advanced mode: Stockfish reviews moves and gives mistakes less weight before ingestion.
- CPU or CUDA training, micro-batches, gradient accumulation and advance preparation of examples.
- A small predictor, approximately 212,000 parameters, with INT8 quantized weight export.
- Interactive `python-chess` SVG board, clock, color selection and playing profiles.
- PGN analyzer with a board, alternatives, a white/black evaluation bar and HTML, PGN and CSV reports.
- Export for CPU console play with a compatible Stockfish installation.

## Getting started in Colab

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LogicGrove/jade-chess-ai/blob/main/Jade_Colab_2_2_1.ipynb)

1. Open [Jade_Colab_2_2_1.ipynb](../Jade_Colab_2_2_1.ipynb). You can download it and upload it to [Google Colab](https://colab.research.google.com/).
2. Run **1 → 2 → 3 → 3B → 4**. In 2, choose the time control and storage settings.
3. To play, run **6**. Jade can work without a trained network by using its fallback hybrid opponent.
4. To ingest data, run **5**. Start with small batches.
5. To train, run **9**. To evaluate the test set, run **10**.
6. Save with **7** before finishing. PGN analysis is in **8**, and saved reports are in **8B**.

The notebook contains the complete program: you do not need to upload the `.py` files separately. Run the startup cells again in each new session. **3B also restores v2 backups: it does not convert an already converted memory again.** Use only one active notebook per memory.

Direct link to the notebook in Colab:

```text
https://colab.research.google.com/github/LogicGrove/jade-chess-ai/blob/main/Jade_Colab_2_2_1.ipynb
```

### CPU and GPU

| Task | Resource |
|---|---|
| Stockfish and advanced review | CPU, not CUDA |
| Neural network training | CPU or a compatible CUDA GPU, such as T4 |
| Exported predictor and console play | CPU, with no VRAM needed for the predictor |
| Storage | Local disk during use; backups and models in Drive when enabled |

There is no switch that turns Stockfish into a GPU engine. The GPU accelerates the network, not Stockfish analysis. INT8 precision reduces the size of the weights, but does not guarantee native INT8 acceleration on every device.

Review the [current Colab terms](https://research.google.com/colaboratory/faq.html). Its FAQ explicitly restricts chess training in free environments without a positive compute unit balance. GPU availability, limits and session duration are not guaranteed. This project does not include mechanisms to bypass them.

## Data and learning

The code reads remote batches from [Lichess/standard-chess-games](https://huggingface.co/datasets/Lichess/standard-chess-games). You do not need to store all Parquet files in Drive. The year and month select the source; rapid and blitz use separate memories and models.

The first 100 half-moves of each game are stored, along with ratings and clocks when available. Sequences are split approximately **80% training, 10% validation and 10% test**, using a content fingerprint. Identical sequences do not cross splits. Shared players and openings can appear in several splits: this is not a player-based split.

Validation and test data do not feed the move frequencies or neural network training. Migrated historical statistics are not used as an independent evaluation set. Lichess ratings use Glicko-2 and are not automatically equivalent to FIDE Elo.

**The repository does not include private games, memory databases or pretrained weights.** Ingesting data in 5 is not the same as training the network: training happens in 9. Jade does not automatically learn from the games you play against it.

### Advanced teaching

In 5, `MODO_ENSENANZA_AVANZADA` reviews retained moves from eligible profiles before ingestion. Stockfish runs without an artificial Elo limit, using a short search: weakening the evaluator does not improve its labels.

| Estimated loss relative to the best option found | Weight |
|---|---:|
| 0–40 centipawns | 1 |
| >40–80 | 0.6 |
| >80–140 | 0.25 |
| >140–250 | 0.05 |
| >250 | 0, except for the optional training sample |

With the default setting, approximately 2% of serious training mistakes are retained with a weight of 0.05. **This does not mean that Jade makes mistakes 2% of the time.** Detected mating errors are excluded; validation does not use this sampling rule. Short searches can be wrong.

The objective is still to imitate human moves weighted by quality: **this is not reinforcement learning or automatic replacement of labels with Stockfish moves**. It does not retrospectively reanalyze the entire historical memory.

On a new installation, first train the original predictor in 9 with `USAR_REVISION_CALIDAD=False`. You can then prepare reviews and train with `USAR_REVISION_CALIDAD=True`; quality fine-tuning needs that initial checkpoint. Start with 100 advanced games and consult [the detailed guide](Jade_2_2_LEEME.en.md).

### Understanding the metrics

| Metric | Desired direction | What it measures |
|---|---|---|
| NLL / prediction error | Lower | Probability assigned to the observed human moves |
| Top-1 | Higher | The human move is the most likely prediction |
| Top-3 | Higher | The human move is among the three most likely predictions |
| `n` | Context, not quality by itself | Number of evaluated positions |

Better human move prediction does not, by itself, demonstrate greater playing strength. Compare metrics with the same dataset and objective; do not directly compare weighted validation with an unweighted human test set. Use validation to select models and reserve the test set for final evaluations, without repeatedly tuning parameters against it.

## Saving and privacy

By default, with Drive enabled:

- Persistent folder: `/content/drive/MyDrive/Jade`.
- Models: `modelos/rapid` or `modelos/blitz`; quality fine-tuning in `calidad_22`.
- Backups: `copias_v2/*.sqlite.zst`, keeping three per time control.
- Reports: `analisis`; user games: `partidas`.
- Working database: `/content/Jade`, temporary and uncompressed.

GitHub stores **code**; it is not the backup of your corpus. Do not publish your entire Drive, keys, tokens, personal games or checkpoints. Although player names are not stored in the corpus, IDs can link to public games: the data should not be described as completely anonymous.

## Repository layout

The notebook for trying Jade stays in the root. `src/` contains Python sources and generated programs; `scripts/` contains the builder and its template; `tests/` contains development checks; `docs/` contains the full documentation; `assets/` contains the branding. Run the commands below from the repository root.

## Project code

| File | Purpose |
|---|---|
| `Jade_Colab_2_2_1.ipynb` | Self-contained notebook for users |
| `src/Jade_Programa.py` | Generated complete program |
| `src/jade_core.py`, `src/jade_storage_v3.py` | Core, data, migration and backups |
| `src/jade_policy.py`, `src/jade_reports.py` | Network, training and metrics |
| `src/jade_quality.py`, `src/jade_advanced.py` | Quality review and weighting |
| `src/jade_engine_v2.py`, `src/jade_hybrid_v3.py` | Hybrid engine and tactical checks |
| `src/jade_ui.py`, `src/jade_pgn.py` | Interface and PGN analyzer |
| `src/Jade_Ligero.py` | CPU console version |
| `scripts/build_jade_v3.py`, `scripts/templates/Jade_v12_base.ipynb` | Builder and historical template; their names do not indicate the published version |
| `tests/verify_*.py` | Focused development checks |

Read [CONTRIBUTING.md](CONTRIBUTING.md) before editing sources or regenerating the notebook. `requirements.txt` documents notebook dependencies; it does not install Stockfish or reinstall PyTorch.

## CPU console

First export an approved model and the memory using cell 11. Install a version of Stockfish compatible with your system and the lightweight dependencies:

```bash
python -m pip install -r requirements-cpu.txt
python src/Jade_Ligero.py --stockfish /ruta/a/stockfish --memory /ruta/jade2_rapid.sqlite --model-dir /ruta/modelo --profile 1300
```

Replace the example paths with your own. The model folder needs `metadata.json` and the corresponding exported weights. Add `--black` to play as Black. This is not an Android application, and equal performance across all devices is not promised.

## License and credits

Copyright (C) 2026 Manu and Jade contributors, for their contributions to the project.

Jade's own code and documentation are distributed under **GNU GPL version 3 or, at your option, any later version** (`GPL-3.0-or-later`). See [LICENSE](../LICENSE). The software is provided without warranty. Dependencies, Stockfish and the data retain their own licenses, described in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

Thanks to the developers of Stockfish, python-chess, Lichess, PyTorch and the other tools used. This publication does not include third-party binaries or their databases.

## Limitations and next steps

Jade is an educational prototype, not a competitive engine or an exact replica of a human player. Remaining work includes measuring strength through controlled games, calibrating profiles, evaluating generalization beyond the corpus and comparing quantization in each target environment. No reproducible results from your private models or response time guarantees are included.

To report a bug, open an Issue with the version, cell, environment and traceback, without personal data. Further instructions are in [CONTRIBUTING.md](CONTRIBUTING.md).
