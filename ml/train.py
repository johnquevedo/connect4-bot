# ml/train.py
"""
AlphaZero-lite training loop:
- Self-play using MCTS guided by current net
- Collect (state_planes, pi, z)
- Train CNN (policy + value)
This is a compact reference implementation intended for learning & experimentation.
"""
from __future__ import annotations
import argparse, random, math, os, time
from typing import List, Tuple
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
except Exception:
    torch = None

from connect4.board import Board, ROWS, COLS, EMPTY, P1, P2
from engine.mcts import MCTS, NNEvaluator
from ml.model import Connect4Net

def encode_board(b: Board):
    # (2,6,7) float32
    p = b.player
    x = [[[(1 if b.grid[r][c]==p else 0) for c in range(COLS)] for r in range(ROWS)],
         [[(1 if b.grid[r][c]==-p else 0) for c in range(COLS)] for r in range(ROWS)]]
    return torch.tensor(x, dtype=torch.float32)

class Memory(Dataset):
    def __init__(self, samples: List[Tuple[torch.Tensor, torch.Tensor, float]]):
        self.data = samples
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        s, pi, z = self.data[idx]
        return s, pi, torch.tensor([z], dtype=torch.float32)

def self_play_game(mcts: MCTS, tau_moves: int = 8) -> List[Tuple[torch.Tensor, List[float], int]]:
    b = Board()
    trace = []
    while not b.terminal():
        pi = mcts.run(b, n_simulations=mcts._n_sims)  # uses attribute set outside
        # temperature
        if len(trace) < tau_moves:
            # sample move from pi
            moves = b.valid_moves()
            r = random.random()
            cum = 0.0
            chosen = moves[0]
            for a in range(COLS):
                cum += pi[a]
                if r <= cum:
                    chosen = a
                    break
        else:
            chosen = max(range(COLS), key=lambda a: pi[a])
            if chosen not in b.valid_moves():
                chosen = random.choice(b.valid_moves())
        state_tensor = encode_board(b)
        trace.append((state_tensor, pi, b.player))
        b.play(chosen)

    winner = b.check_winner()
    z_for_p1 = 0
    if winner == P1: z_for_p1 = 1
    elif winner == P2: z_for_p1 = -1

    # assign +1/-1 from each state's player perspective
    results = []
    for s, pi, player in trace:
        z = z_for_p1 if player == P1 else -z_for_p1
        results.append((s, pi, z))
    return results

def train_loop(args):
    if torch is None:
        raise RuntimeError("Install torch to train: pip install torch")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = Connect4Net().to(device)
    optim = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.l2)

    samples = []
    for it in range(args.iters):
        # Self-play
        evaluator = NNEvaluator(net, device=device)
        mcts = MCTS(evaluator=evaluator, c_puct=args.c_puct)
        mcts._n_sims = args.sims  # store on object for self_play_game
        for g in range(args.games):
            samples.extend(self_play_game(mcts))

        # Train
        mem = Memory([(s, torch.tensor(pi, dtype=torch.float32), z) for s,pi,z in samples])
        dl = DataLoader(mem, batch_size=args.bs, shuffle=True, drop_last=True)
        for ep in range(args.epochs):
            total = 0.0
            for s, pi, z in dl:
                s = s.to(device)
                pi = pi.to(device)
                z = z.to(device)
                logits, value = net(s)
                policy_loss = F.cross_entropy(logits, pi.argmax(dim=1))
                value_loss = F.mse_loss(value, z)
                loss = policy_loss + args.vloss_coeff * value_loss
                optim.zero_grad()
                loss.backward()
                optim.step()
                total += loss.item()
            print(f"[iter {it+1}/{args.iters}] epoch {ep+1}/{args.epochs} loss={total/len(dl):.4f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(net.state_dict(), args.out)
    print(f"Saved model to {args.out}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=1, help="outer iters: (self-play -> train) cycles")
    ap.add_argument("--games", type=int, default=100, help="self-play games per iter")
    ap.add_argument("--sims", type=int, default=200, help="MCTS simulations per move")
    ap.add_argument("--epochs", type=int, default=5, help="epochs per training stage")
    ap.add_argument("--bs", type=int, default=64, help="batch size")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--l2", type=float, default=1e-4, help="L2 weight decay")
    ap.add_argument("--vloss_coeff", type=float, default=1.0)
    ap.add_argument("--c_puct", type=float, default=1.25)
    ap.add_argument("--out", type=str, default="models/connect4.pt")
    args = ap.parse_args()
    train_loop(args)

if __name__ == "__main__":
    main()
