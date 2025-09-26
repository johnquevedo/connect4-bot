# Connect 4 Bot (Python)

A Connect 4 bot with both **classic search engines** and an **ML-powered MCTS**.  
You can play immediately with the Minimax engine, or train a neural net to guide MCTS.

## Features
- **Negamax + Alpha-Beta + Transposition Table** — strong baseline, no ML required.  
- **Monte Carlo Tree Search (MCTS)** — usable as-is, improves significantly with a trained model.  
- **Tkinter UI** — lightweight interface to play against the bot.  
- **Training script (AlphaZero-lite)** — lets the bot self-play and train a small CNN in PyTorch.  

---

## Installation
It’s best to use a virtual environment:
```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install torch   # only needed for training/NN-guided MCTS
