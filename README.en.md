<p><img src="assets/jade-banner.png" alt="Jade · Chess + AI" width="960"></p>

# Jade Chess AI

[Español](README.md) | **English**

Play chess against an AI that combines Stockfish with human playing styles. Jade is an educational, experimental project by **Manu**. Version **2.2.1**.

## Try Jade

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LogicGrove/jade-chess-ai/blob/main/Jade_Colab_2_2_1.ipynb)

1. **Open the notebook** using the button above and sign in to Google if requested.
2. **Set up the session:** run cells **1 → 2 → 3 → 3B → 4**, in that order. In cell 2, choose the time control and whether to save to Drive.
3. **Play:** run cell **6**, choose your color and a profile, and use the board.
4. **Save:** run cell **7** before finishing if you want to keep the memory with Drive enabled.

**Your first game does not require training a network, downloading data or uploading the `.py` files.** The notebook contains the complete program. You can start on CPU; a GPU is optional for neural training. The notebook interface is currently in Spanish.

In each new session, rerun the setup cells. Use only one active notebook per memory folder. Profiles 1200–1600 are style targets; they are not measured Elo ratings.

## Explore further

- **Analyze a PGN:** cell **8**; saved reports are in **8B**.
- **Train, use the console or modify the code:** [advanced README](docs/README.advanced.en.md). It retains all the previous technical documentation.
- **Download the notebook:** [repository file](Jade_Colab_2_2_1.ipynb) or [release 2.2.1](https://github.com/LogicGrove/jade-chess-ai/releases/tag/v2.2.1).

Before training on Colab, check [its conditions](https://research.google.com/colaboratory/faq.html): chess training is restricted on free environments without a positive compute unit balance. GPU availability and session duration vary.

Something went wrong? Open an Issue with the cell, error and environment, without personal data. [Contributing](docs/CONTRIBUTING.md).

Code and documentation: [GPL-3.0-or-later](LICENSE). [Credits and third-party licenses](docs/THIRD_PARTY_NOTICES.md).
