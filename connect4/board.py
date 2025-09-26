# connect4/board.py
from __future__ import annotations
from typing import List, Optional, Tuple

ROWS, COLS = 6, 7
EMPTY, P1, P2 = 0, 1, -1

class Board:
    """
    Simple 6x7 Connect Four board.
    Player to move is tracked as self.player (1 or -1). Internally stores integers.
    """

    __slots__ = ("grid", "heights", "player", "moves_played", "_winner_cache")

    def __init__(self):
        self.grid = [[EMPTY for _ in range(COLS)] for _ in range(ROWS)]
        self.heights = [0]*COLS  # how many pieces in each column
        self.player = P1         # 1 = red (first player), -1 = yellow (second)
        self.moves_played = 0
        self._winner_cache = None

    def copy(self) -> "Board":
        b = Board()
        b.grid = [row[:] for row in self.grid]
        b.heights = self.heights[:]
        b.player = self.player
        b.moves_played = self.moves_played
        b._winner_cache = self._winner_cache
        return b

    def valid_moves(self) -> List[int]:
        return [c for c in range(COLS) if self.heights[c] < ROWS]

    def play(self, col: int) -> bool:
        """Drop a piece in col for current player. Returns True if success."""
        if col < 0 or col >= COLS or self.heights[col] >= ROWS:
            return False
        r = self.heights[col]
        self.grid[r][col] = self.player
        self.heights[col] += 1
        self.moves_played += 1
        self._winner_cache = None  # invalidate
        self.player = -self.player
        return True

    def undo(self, col: int) -> None:
        r = self.heights[col] - 1
        self.grid[r][col] = EMPTY
        self.heights[col] -= 1
        self.moves_played -= 1
        self._winner_cache = None
        self.player = -self.player

    def is_full(self) -> bool:
        return self.moves_played >= ROWS * COLS

    def check_winner(self) -> int:
        """Returns P1 or P2 if there's a winner, 0 otherwise. Cached."""
        if self._winner_cache is not None:
            return self._winner_cache
        # Horizontal, vertical, diag checks
        g = self.grid
        # horizontal
        for r in range(ROWS):
            for c in range(COLS-3):
                s = g[r][c]
                if s and s == g[r][c+1] == g[r][c+2] == g[r][c+3]:
                    self._winner_cache = s
                    return s
        # vertical
        for c in range(COLS):
            for r in range(ROWS-3):
                s = g[r][c]
                if s and s == g[r+1][c] == g[r+2][c] == g[r+3][c]:
                    self._winner_cache = s
                    return s
        # diag down-right
        for r in range(ROWS-3):
            for c in range(COLS-3):
                s = g[r][c]
                if s and s == g[r+1][c+1] == g[r+2][c+2] == g[r+3][c+3]:
                    self._winner_cache = s
                    return s
        # diag up-right
        for r in range(3, ROWS):
            for c in range(COLS-3):
                s = g[r][c]
                if s and s == g[r-1][c+1] == g[r-2][c+2] == g[r-3][c+3]:
                    self._winner_cache = s
                    return s
        self._winner_cache = 0
        return 0

    def terminal(self) -> bool:
        return self.check_winner() != 0 or self.is_full()

    def key(self) -> Tuple[Tuple[int, ...], ...]:
        """A hashable board key including player to move."""
        # Represent from bottom row to top
        return tuple(tuple(row) for row in self.grid) + (tuple([self.player]),)

    # ---- Heuristic evaluation (for Minimax) ----
    def evaluate_heuristic(self, for_player: int) -> int:
        """
        Cheap linear eval: count open 2/3s, center control.
        Positive is good for `for_player`.
        """
        winner = self.check_winner()
        if winner == for_player:
            return 1_000_000
        if winner == -for_player:
            return -1_000_000

        score = 0
        center_col = COLS // 2
        # Center control
        center_count = sum(1 for r in range(ROWS) if self.grid[r][center_col] == for_player)
        opp_center_count = sum(1 for r in range(ROWS) if self.grid[r][center_col] == -for_player)
        score += 6 * (center_count - opp_center_count)

        # Count windows of length 4
        def window_score(window):
            count_for = window.count(for_player)
            count_opp = window.count(-for_player)
            count_empty = window.count(EMPTY)
            s = 0
            if count_for == 3 and count_empty == 1:
                s += 100
            elif count_for == 2 and count_empty == 2:
                s += 10
            if count_opp == 3 and count_empty == 1:
                s -= 90
            elif count_opp == 2 and count_empty == 2:
                s -= 9
            return s

        # Horizontal
        for r in range(ROWS):
            row = [self.grid[r][c] for c in range(COLS)]
            for c in range(COLS-3):
                score += window_score(row[c:c+4])
        # Vertical
        for c in range(COLS):
            col = [self.grid[r][c] for r in range(ROWS)]
            for r in range(ROWS-3):
                score += window_score(col[r:r+4])
        # Diagonals
        for r in range(ROWS-3):
            for c in range(COLS-3):
                window = [self.grid[r+i][c+i] for i in range(4)]
                score += window_score(window)
        for r in range(3, ROWS):
            for c in range(COLS-3):
                window = [self.grid[r-i][c+i] for i in range(4)]
                score += window_score(window)
        return score

    def printable(self) -> str:
        symbols = {EMPTY: ".", P1: "X", P2: "O"}
        rows = []
        for r in range(ROWS-1, -1, -1):
            rows.append(" ".join(symbols[self.grid[r][c]] for c in range(COLS)))
        return "\n".join(rows)
