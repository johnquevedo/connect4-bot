# ui/play_tk.py
from __future__ import annotations
import tkinter as tk
from tkinter import filedialog, messagebox
import threading
from typing import Optional
from connect4.board import Board, P1, P2, ROWS, COLS
from engine.minimax import find_best_move
from engine.mcts import MCTS, NNEvaluator
import os

# Optional NN
try:
    from ml.model import load_model
    import torch
except Exception:
    load_model = None
    torch = None

CELL = 80
PADDING = 20
BG = "#10264d"
COLOR_EMPTY = "#ebf2ff"
COLOR_P1 = "#ff595e"
COLOR_P2 = "#1982c4"

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Connect 4 — Minimax / MCTS")
        self.resizable(False, False)

        self.board = Board()
        self.canvas = tk.Canvas(self, width=COLS*CELL+2*PADDING, height=ROWS*CELL+2*PADDING, bg=BG, highlightthickness=0)
        self.canvas.grid(row=0, column=0, columnspan=6, padx=10, pady=10)

        # Controls
        tk.Label(self, text="Engine:").grid(row=1, column=0, sticky="e")
        self.engine_var = tk.StringVar(value="Minimax")
        tk.OptionMenu(self, self.engine_var, "Minimax", "MCTS", "MCTS (NN)").grid(row=1, column=1, sticky="w")

        tk.Label(self, text="Depth/Sims:").grid(row=1, column=2, sticky="e")
        self.depth_var = tk.IntVar(value=7)
        tk.Entry(self, textvariable=self.depth_var, width=5).grid(row=1, column=3, sticky="w")

        tk.Label(self, text="Human:").grid(row=1, column=4, sticky="e")
        self.human_var = tk.StringVar(value="P1 (X)")
        tk.OptionMenu(self, self.human_var, "P1 (X)", "P2 (O)").grid(row=1, column=5, sticky="w")

        self.btn_new = tk.Button(self, text="New Game", command=self.new_game)
        self.btn_new.grid(row=2, column=0, pady=5)

        self.btn_load = tk.Button(self, text="Load NN", command=self.load_nn)
        self.btn_load.grid(row=2, column=1, pady=5)

        self.status = tk.StringVar(value="Click a column to play.")
        tk.Label(self, textvariable=self.status).grid(row=2, column=2, columnspan=4, sticky="w")

        self.canvas.bind("<Button-1>", self.on_click)

        self.nn_model = None
        self.ai_th = None

        self.draw_board()

    def new_game(self):
        self.board = Board()
        self.draw_board()
        self.status.set("New game!")
        self.after(100, self.maybe_ai_move)

    def load_nn(self):
        if load_model is None:
            messagebox.showinfo("Info", "PyTorch not installed. Run: pip install torch")
            return
        path = filedialog.askopenfilename(initialdir=os.getcwd(), title="Pick a .pt model", filetypes=[("PyTorch model", "*.pt")])
        if not path:
            return
        device = "cuda" if (torch and torch.cuda.is_available()) else "cpu"
        try:
            self.nn_model = load_model(path, device=device)
            messagebox.showinfo("Loaded", f"Loaded model: {os.path.basename(path)}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def draw_board(self):
        self.canvas.delete("all")
        w, h = COLS*CELL+2*PADDING, ROWS*CELL+2*PADDING
        # Board holes
        for r in range(ROWS):
            for c in range(COLS):
                x0 = PADDING + c*CELL + 5
                y0 = PADDING + (ROWS-1-r)*CELL + 5
                x1, y1 = x0 + CELL - 10, y0 + CELL - 10
                val = self.board.grid[r][c]
                color = COLOR_EMPTY if val == 0 else (COLOR_P1 if val == P1 else COLOR_P2)
                self.canvas.create_oval(x0, y0, x1, y1, fill=color, outline=BG, width=2)

        # Column hover indicators (numbers)
        for c in range(COLS):
            x = PADDING + c*CELL + CELL//2
            y = PADDING//2
            self.canvas.create_text(x, y, text=str(c), fill="white")

    def on_click(self, event):
        col = (event.x - PADDING) // CELL
        if col < 0 or col >= COLS: 
            return
        if self.board.terminal():
            return
        human_is_p1 = self.human_var.get().startswith("P1")
        human_player = P1 if human_is_p1 else P2
        if self.board.player != human_player:
            return
        if not self.board.play(col):
            return
        self.draw_board()
        self.after(50, self.maybe_ai_move)

    def maybe_ai_move(self):
        if self.board.terminal():
            self.end_if_done()
            return
        human_is_p1 = self.human_var.get().startswith("P1")
        ai_player = P2 if human_is_p1 else P1
        if self.board.player != ai_player:
            return
        # run AI on a thread
        if self.ai_th and self.ai_th.is_alive():
            return
        self.ai_th = threading.Thread(target=self._ai_move_once, daemon=True)
        self.ai_th.start()

    def _ai_move_once(self):
        engine = self.engine_var.get()
        param = max(1, int(self.depth_var.get()))
        move = None
        if engine == "Minimax":
            move = find_best_move(self.board.copy(), depth=param)
        elif engine.startswith("MCTS"):
            evaluator = None
            if engine == "MCTS (NN)" and self.nn_model is not None:
                evaluator = NNEvaluator(self.nn_model)
            mcts = MCTS(evaluator=evaluator)
            move = mcts.best_move(self.board.copy())
        else:
            move = find_best_move(self.board.copy(), depth=5)
        if move is not None and move >= 0:
            self.board.play(move)
        self.after(0, self.post_ai_move)

    def post_ai_move(self):
        self.draw_board()
        self.end_if_done()

    def end_if_done(self):
        if self.board.terminal():
            w = self.board.check_winner()
            if w == P1:
                self.status.set("X wins!")
            elif w == P2:
                self.status.set("O wins!")
            else:
                self.status.set("Draw.")
        else:
            self.status.set("Your turn!" if (self.board.player == (P1 if self.human_var.get().startswith('P1') else P2)) else "Thinking...")

def launch():
    app = App()
    app.mainloop()

if __name__ == "__main__":
    launch()
