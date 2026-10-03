# Jade 2.2.1: optional advanced teaching

[Español](Jade_2_2_LEEME.md) | **English** · [Main README](../README.en.md)

This update preserves the v2 memory, games and 2.0/2.1 models.
It does not contain your data or weights: it restores them from your existing Drive folder.
It does not convert a v2 memory again or require downloading the games again.

## New: review games during ingestion, in cell 5

The **MODO_ENSENANZA_AVANZADA** checkbox makes Stockfish review each new training
or validation game before ingestion. It covers the first 100 stored half-moves,
and only turns belonging to eligible profiles (1200–1600). Forced moves receive
weight 1 without a search. It does not review the test set to learn from it.

Stockfish runs without an Elo limit, with a short search budget. Lowering its Elo
makes it deliberately choose inferior moves and does not make it a good evaluator
of human move quality. Jade retains both players' ratings as network context,
along with its playing profiles. This weighting does not calibrate a human Elo rating.

To update from 2.2:

1. Save the memory with 7 and wait for any training in the old notebook to finish.
2. Open the new notebook and run **1 → 2 → 3 → 3B → 4**, using the same folder and time control.
   3B restores the v2 backup when needed; it does not convert it again.
3. In **5**, enable advanced mode and start with these settings:

| Field | Initial value |
|---|---:|
| MODO_ENSENANZA_AVANZADA | Enabled |
| PARTIDAS_AVANZADAS | 100 |
| SEGUNDOS_ANALISIS_AVANZADO | 0.08 |
| ERRORES_GRAVES_CONSERVADOS_PCT | 2 |

PARTIDAS_AVANZADAS limits the accepted batch, including reserved splits;
NUEVAS_PARTIDAS from cell 2 is also respected when smaller. This does not take
100 games from each profile. Games rejected by time control, rating or duplication
are not analyzed.

4. Run **9** with **USAR_REVISION_CALIDAD=True**, RONDAS=2, MICRO_LOTE=0 and
   LOTE_EFECTIVO=64. It continues the corrected model if one exists; otherwise,
   it starts from your original weights. Cell 5 updates frequencies and prepares
   labels; it does not train the network.
5. If validation examples are missing, add another batch or use 9A with 150 validation
   games and 6 positions per game. The model is activated only if it passes its
   validation checks; 100 initial games do not guarantee approval.

### How each move is weighted

| Short evaluation against the best option found | Weight |
|---|---:|
| Loss of 0–40 centipawns | 1 |
| More than 40 and up to 80 cp | 0.6 |
| More than 80 and up to 140 cp | 0.25 |
| More than 140 and up to 250 cp | 0.05 |
| More than 250 cp | 0, except for the small training sample |
| Allows an avoidable mate or misses a detected mate | 0 |
| All alternatives lead to being checkmated | 0.25 |

With the setting 2, approximately **2% of training mistakes losing more than 250 cp**
are retained with **weight 0.05**. The others have zero weight. The sample is
deterministic: a move is not sampled again in every epoch. Mating errors are
excluded, and validation does not use this sampling rule. Set it to 0 to exclude
all such serious mistakes. This is not a probability of making a mistake during play.

Weights affect new frequencies and reviewed training. The actual move remains
the label: it is not replaced with a Stockfish move, and this is not reinforcement
learning. Historical frequencies and the original corpus remain; enabling the
checkbox does not retrospectively correct the two million games.
The tactical protection from 2.2 still constrains playing decisions.

### Duration, interruptions and storage

This is considerably slower than ingesting Parquet without analysis. Each position
may need a search, a second comparison and a triple check when a serious mistake
is detected. The budget is limited by time and nodes, whichever is reached first.
Stockfish uses up to two CPU threads and 64 MB of hash; it does not use the GPU.
For example, 100 games with 60 reviewable positions each amount to 6,000 positions.
At 0.08 s per search and two searches per position, that would be about 16 minutes
of computation before extra checks and other tasks; reaching the node limit may
reduce this. This is an illustration, not a measurement of your session. The log
shows actual progress.

Ingestion commits the game, weights and cursor in a single transaction. An
interruption halfway through a game does not duplicate statistics. Partial analysis
saves completed moves and is backed up approximately every 90 s, and when stopping
with the Colab button if KeyboardInterrupt is received. Memory backups are made
between games approximately every 120 s and on exit. An abrupt crash may lose work
since the last backup. To resume, run the startup cells and then 5 again with
advanced mode enabled. This does not keep Colab connected or bypass its limits.

New parameters affect games that have not yet been reviewed. A game already under
review finishes with its original parameters so that a disconnection does not
change its weights. Duplicates are not analyzed again.

Committed labels are included in the memory database and in the auxiliary backup
`Jade/modelos/rapid/revision_calidad` (or `blitz`). If that auxiliary backup is
missing, it is reconstructed from memory before training. The partial cache is
deleted when a game is committed. The model remains in `calidad_22` to preserve
continuity with 2.2. CPU export omits the corpus and labels, retaining weighted frequencies.

Disabling the checkbox in 5 restores normal ingestion. Training in 9 without
USAR_REVISION_CALIDAD uses raw moves again, including mistakes: for this objective,
keep review enabled. Cell 10 retains its original human test set.

2.2.1 checks: ingestion with local Parquet, fractional weights used by the engine,
test exclusion, deduplication, interruption midway through a game, restoration from
both backups and label reconstruction. CPU training and real Stockfish tests for
mates from Black's perspective are also run. Your private games and models have
not been used; measuring actual quality on your games remains pending.

Official reference on limited strength and CPU usage:
https://official-stockfish.github.io/docs/stockfish-wiki/Stockfish-FAQ.html

## What was corrected

The previous version trained to imitate the human move, whether good or bad.
The mixture of frequencies and network predictions also tolerated high losses,
especially in openings. A large frequency could dominate the decision even when
the move was weak. Higher agreement with humans on the test set does not demonstrate
greater strength of the hybrid opponent.

1. **Shared final check.** All proposals, including network proposals, pass through
   an estimated loss limit of 65–140 centipawns depending on the profile.
   Losing an advantage is also mildly penalized within that margin. A mistake
   is never forced. These limits are heuristics, not Elo calibration.
   The combined probability of candidates losing more than 60 cp is capped at
   3–8%, depending on the profile; there is no requirement to reach that percentage.
   The cap depends on available evaluations and does not guarantee that percentage
   in actual games.
2. **Reference and tactical verification.** A single-variation search is performed;
   if another move is selected, it is checked against the reference with two variations.
   Detected mates are preserved, and allowing mate is rejected when an alternative exists.
   Computation may increase to approximately 1.8 times the previous budget;
   the clock deducts actual computation and the pause. A short analysis can still fail.
3. **Weighted supervised fine-tuning.** The new cell 9A reviews a sample with
   Stockfish. Cell 9 learns from that sample using quality weights. This is not RL.
   The human move remains the label; it is not replaced with the engine's first choice.
4. **PGN viewer.** Animated vertical black/white bar, numerical evaluation in cp
   and pawns, separate mate values, and buttons for before/after and the engine alternative.
   +100 cp equals +1.00 pawns from White's perspective. The height is a nonlinear
   visual scale, not a win percentage. Old HTML reports do not update themselves.

## Updating and playing with what was learned

1. If the previous notebook is still active, finish the task and save the memory with 7.
   Weights are saved during training; 7 saves the memory, it does not train.
2. Download the updated `Jade_Colab_2_2_1.ipynb` and open it in Colab as a new copy.
   Keep the previous notebook as a reference and use only one active notebook
   per memory/model folder.
3. In 2, select the same account, Drive folder and time control (`rapid` or `blitz`).
4. Run **1 → 2 → 3 → 3B → 4 → 6**. 3B restores the v2 backup; it should not repeat
   the old conversion. Cell 4 announces Jade 2.2's tactical protection.
5. Try the opponent before fine-tuning the network: the check already applies
   to your current weights.

The update does not delete or modify your historical games, so that the original
training can be reproduced. It does not guarantee error-free play or an actual Elo
rating. Without analyzing your PGN, each specific failure cannot be attributed
to a single cause.

## Also improving learning

Run **9A** with the initial values:

| Setting | Value |
|---|---:|
| New training games to review | 250 |
| Reviewed validation games, total target | 150 |
| Positions per game | 6 |
| Seconds per search | 0.12 |

Review uses the CPU and may take several minutes. Each position may require
more than one search; mistakes that would be excluded are checked with additional
computation. It does not automatically analyze the two million games. Running 9A
again adds another training batch and reuses validation. A partially reviewed
game may be repeated when resuming; completed games are not duplicated.

Then run **9** with `USAR_REVISION_CALIDAD=True`, **RONDAS=2**,
`PARTIDAS_POR_RONDA=2000`, `MICRO_LOTE=0`, `LOTE_EFECTIVO=64` and device `auto`.
2000 is a maximum: if only 250 reviewed games are available, it uses those 250.
Position selection comes from 9A; `POSICIONES_POR_PARTIDA` in 9 does not expand it.
Avoid repeating hundreds of rounds on such a small sample: expand 9A and check
validation. The T4 does not accelerate Stockfish analysis.

| Estimated loss of the human move | Weight for network fine-tuning |
|---|---:|
| 0–40 cp | 1 |
| More than 40, up to 80 cp | 0.6 |
| More than 80, up to 140 cp | 0.25 |
| More than 140, up to 250 cp | 0.05 |
| More than 250 cp, allows avoidable mate or misses mate | 0 (excluded) |

When all alternatives lead to being checkmated, the move is not blamed for an
inevitable result: it receives weight 0.25. The thresholds and finite search may
undervalue a correct sacrifice. Increasing the time in 9A improves review of new
samples, but does not automatically reanalyze those already stored.

The corrected network is saved in `Jade/modelos/rapid/calidad_22` (or `blitz`).
The original checkpoint `Jade/modelos/rapid/training.pt` is preserved. When
fine-tuning starts for the first time, it copies the original weights and optimizer
and uses a lower learning rate. The new model must beat its reference and the
active model on weighted validation before activation. Models are not selected
using the test set.

Labels are backed up in `Jade/modelos/rapid/revision_calidad`, with a previous
backup retained. Local SQLite is used during the session; training does not query
an open database directly on Drive.

After training, run 6. To return to the original human predictor for the session:

```python
jade.policy = load_jade_policy(MODEL_DIR, RITMO)
```

The tactical check remains active. You do not need to delete files.

## Measuring results honestly

- Quality validation uses different weights and is not numerically comparable
  to the original test set. The report explicitly identifies it.
- Cell 10 keeps the unfiltered human test set. Top-1 may decrease when certain
  mistakes are no longer imitated. Do not use the test set to tune thresholds
  or select models.
- The test evaluates the network, not the final mixture with Stockfish. To study
  actual mistakes by the opponent, save complete PGNs and analyze them with more
  time in cell 8.
- Profile 1300 remains a style target, not a certified Elo rating.

## Compatibility and scope of testing

The neural architecture and v2 memory schema are preserved. Checks cover
weighting, mates, normalization, corpus separation, backup/restoration,
CPU continuation from previous weights and notebook syntax. Real Stockfish 17.1,
mates from both colors, incremental review and PGNs containing mate are also tested.
The viewer is checked in Chromium at widths of 360 and 1200 pixels, including
navigation and bar updates. Your weights and two million games are not available
to measure an actual increase in strength. CUDA/FP16, Colab and synchronization
with your Drive need to be checked in your own session.

The provider's usage policy still applies: Colab publishes restrictions on chess
training in its free tier without a compute unit balance.
Consult https://research.google.com/colaboratory/faq.html before long sessions.

Technical references:

- https://python-chess.readthedocs.io/en/latest/engine.html
- https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
