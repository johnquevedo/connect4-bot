# engine/minimax.py
from __future__ import annotations
import math, time
from typing import Dict, Tuple, Optional
from connect4.board import Board, P1, P2

INF = 10**9

class Transposition:
    def __init__(self):
        self.table: Dict[Tuple, Tuple[int,int]] = {}  # key -> (depth, score)

    def get(self, key: Tuple, depth: int) -> Optional[int]:
        if key in self.table:
            d, v = self.table[key]
            if d >= depth:
                return v
        return None

    def put(self, key: Tuple, depth: int, value: int):
        self.table[key] = (depth, value)

def order_moves(b: Board, player: int):
    # Prefer center moves for better pruning
    center = len(b.heights)//2
    moves = b.valid_moves()
    return sorted(moves, key=lambda c: abs(c - center))

def negamax(b: Board, depth: int, alpha: int, beta: int, player: int, tt: Transposition) -> int:
    winner = b.check_winner()
    if winner == player:
        return 1_000_000
    elif winner == -player:
        return -1_000_000
    if depth == 0 or b.is_full():
        return b.evaluate_heuristic(player)

    key = b.key()
    cached = tt.get(key, depth)
    if cached is not None:
        return cached

    best = -INF
    for c in order_moves(b, player):
        if not b.play(c): 
            continue
        val = -negamax(b, depth-1, -beta, -alpha, -player, tt)
        b.undo(c)
        if val > best:
            best = val
        if best > alpha:
            alpha = best
        if alpha >= beta:
            break  # beta cutoff

    tt.put(key, depth, best)
    return best

def find_best_move(b: Board, depth: int) -> int:
    tt = Transposition()
    player = b.player
    best_move = None
    best_score = -INF
    for c in order_moves(b, player):
        if not b.play(c):
            continue
        score = -negamax(b, depth-1, -INF, INF, -player, tt)
        b.undo(c)
        if score > best_score:
            best_score, best_move = score, c
    return best_move if best_move is not None else (b.valid_moves()[0] if b.valid_moves() else -1)
