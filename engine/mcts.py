# engine/mcts.py
from __future__ import annotations
import math, random
from typing import Dict, Optional, Tuple, List
from connect4.board import Board, P1, P2, EMPTY, ROWS, COLS

try:
    import torch
    import torch.nn.functional as F
except Exception:
    torch = None
    F = None

class NNEvaluator:
    """
    Optional neural net wrapper.
    Expects a PyTorch module with forward(board_tensor) -> (policy_logits[7], value_scalar).
    """
    def __init__(self, model: Optional["torch.nn.Module"]=None, device: str="cpu"):
        self.model = model
        self.device = device
        if model is not None and torch is None:
            raise RuntimeError("PyTorch not available but model provided.")

    def encode(self, b: Board):
        # Shape (2, 6, 7): planes for current player and opponent
        p = b.player
        x = [[[(1 if b.grid[r][c]==p else 0) for c in range(COLS)] for r in range(ROWS)],
             [[(1 if b.grid[r][c]==-p else 0) for c in range(COLS)] for r in range(ROWS)]]
        if torch is None:
            return x  # for debugging
        t = torch.tensor(x, dtype=torch.float32, device=self.device).unsqueeze(0)  # (1,2,6,7)
        return t

    def __call__(self, b: Board):
        valid = b.valid_moves()
        if self.model is None or torch is None:
            # Uniform policy over valid moves, value 0
            priors = [0.0]*COLS
            for m in valid:
                priors[m] = 1.0/len(valid)
            return priors, 0.0
        with torch.no_grad():
            t = self.encode(b)
            logits, value = self.model(t)
            logits = logits.squeeze(0)  # (7,)
            value = value.squeeze().item()
            # Mask invalid moves then softmax
            mask = torch.full_like(logits, float("-inf"))
            for m in valid:
                mask[m] = 0.0
            probs = F.softmax(logits + mask, dim=0).cpu().tolist()
            return probs, value

class MCTS:
    def __init__(self, evaluator: Optional[NNEvaluator]=None, c_puct: float=1.25):
        self.Q: Dict[Tuple, Dict[int,float]] = {}  # Q[s][a]
        self.N: Dict[Tuple, Dict[int,int]] = {}    # N[s][a]
        self.P: Dict[Tuple, Dict[int,float]] = {}  # prior
        self.evaluator = evaluator or NNEvaluator(None)
        self.c_puct = c_puct

    def ucb(self, s_key: Tuple, a: int) -> float:
        q = self.Q[s_key].get(a, 0.0)
        n_sa = self.N[s_key].get(a, 0)
        n_s = sum(self.N[s_key].values()) + 1e-8
        p = self.P[s_key].get(a, 0.0)
        u = self.c_puct * p * math.sqrt(n_s) / (1 + n_sa)
        return q + u

    def select(self, b: Board) -> List[int]:
        path = []
        while True:
            s_key = b.key()
            if s_key not in self.P:
                self.expand(b)
                return path
            # choose a with max UCB among valid moves
            best_a, best_ucb = None, float("-inf")
            for a in b.valid_moves():
                u = self.ucb(s_key, a)
                if u > best_ucb:
                    best_ucb, best_a = u, a
            path.append(best_a)
            b.play(best_a)
            if b.terminal():
                return path

    def expand(self, b: Board):
        s_key = b.key()
        priors, _ = self.evaluator(b)
        self.P[s_key] = {}
        self.Q[s_key] = {}
        self.N[s_key] = {}
        for a in b.valid_moves():
            self.P[s_key][a] = priors[a]
            self.N[s_key][a] = 0
            self.Q[s_key][a] = 0.0

    def simulate(self, b: Board) -> float:
        # Returns value from perspective of starting player at node
        # After selection, if leaf: evaluate value with NN; if terminal: +/-1/0
        start_player = b.player
        winner = b.check_winner()
        if winner != 0:
            return 1.0 if winner == -start_player else -1.0  # because b.player already switched after last move? careful
        if b.is_full():
            return 0.0
        s_key = b.key()
        if s_key not in self.P:
            # Leaf
            priors, value = self.evaluator(b)
            self.P[s_key] = {a: priors[a] for a in b.valid_moves()}
            self.Q[s_key] = {a: 0.0 for a in b.valid_moves()}
            self.N[s_key] = {a: 0 for a in b.valid_moves()}
            return value
        # Otherwise select best child and recurse
        best_a, best_ucb = None, float("-inf")
        for a in b.valid_moves():
            u = self.ucb(s_key, a)
            if u > best_ucb:
                best_ucb, best_a = u, a
        b.play(best_a)
        v = -self.simulate(b)  # perspective switch
        b.undo(best_a)
        # Backup
        q = self.Q[s_key].get(best_a, 0.0)
        n = self.N[s_key].get(best_a, 0)
        new_q = (n * q + v) / (n + 1)
        self.Q[s_key][best_a] = new_q
        self.N[s_key][best_a] = n + 1
        return v

    def run(self, b: Board, n_simulations: int = 200) -> List[float]:
        for _ in range(n_simulations):
            bb = b.copy()
            self.simulate(bb)
        s_key = b.key()
        counts = [self.N.get(s_key, {}).get(a, 0) for a in range(COLS)]
        total = sum(counts)
        if total == 0:
            # choose random valid
            policy = [0.0]*COLS
            vm = b.valid_moves()
            if vm:
                for a in vm: policy[a] = 1/len(vm)
            return policy
        # Return visit count distribution
        return [c/total for c in counts]

    def best_move(self, b: Board, temperature: float = 0.0) -> int:
        pi = self.run(b)
        vm = b.valid_moves()
        if not vm:
            return -1
        if temperature <= 1e-6:
            # argmax
            best = max(vm, key=lambda a: pi[a])
            return best
        # sample
        r = random.random()
        cum = 0.0
        for a in range(COLS):
            cum += pi[a]
            if r <= cum:
                return a
        return vm[0]
