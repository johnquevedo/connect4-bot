# Connect 4 Bot (Minimax + MCTS + Optional ML) — Python

This project gives you:
- A *strong baseline* engine using **Negamax + Alpha-Beta + transposition table** (no ML).
- A **Monte Carlo Tree Search (MCTS)** engine that can optionally be guided by a policy/value neural net.
- A **Tkinter UI** so you can play against the bot right away.
- A **training script** (AlphaZero-lite) to self-play and train a small CNN with PyTorch (optional, only if you want ML).

> You can immediately play using the Minimax engine (no ML needed). When you later train a model, the MCTS engine can use it.

## 0) Install (recommended virtualenv)
```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install torch  # only if you plan to train/use the NN-guided MCTS
```
*(UI uses only stdlib Tkinter.)*

## 1) Run the UI
```bash
python main.py
```
- Choose engine: **Minimax** (fast/strong out of the box) or **MCTS** (works now with random priors; gets much stronger with a trained net).
- Choose who starts, depth/sims, and press **New Game**.

## 2) (Optional) Train a model
```bash
python -m ml.train --games 200 --sims 200 --epochs 5 --out models/connect4.pt
```
Then in the UI, select **MCTS (NN)** and point to that weights file.

### Training overview
- **Self-play** with MCTS generates (state, policy-target, value) triplets.
- A compact **CNN** learns policy & value. Loss = CE(policy) + MSE(value) + L2.
- You can iterate: train → play → train more.

## 3) Project structure
```
connect4_ml_bot/
├─ main.py                  # Launches Tkinter UI
├─ ui/play_tk.py            # UI code
├─ connect4/board.py        # Rules & board representation
├─ engine/minimax.py        # Negamax + Alpha-beta + TT
├─ engine/mcts.py           # MCTS (UCT), optional NN guidance
├─ ml/model.py              # Small PyTorch CNN (policy + value)
├─ ml/train.py              # Self-play + training loop
└─ models/                  # Saved .pt models land here
```

## 4) Tips
- Minimax depth 6–8 is quite challenging (and quick). Depth 10+ can be slow.
- MCTS strength scales with `--sims`. With a trained net, 400–1600 sims feels very strong.
- Connect 4 is *solved*, but our ML approach targets *strong play* without a hard-coded DB.

Have fun! 🎮
