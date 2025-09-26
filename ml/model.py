# ml/model.py
from __future__ import annotations
from typing import Tuple
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception as e:
    torch = None
    nn = None
    F = None

class Connect4Net(torch.nn.Module if torch else object):
    """
    Tiny CNN for Connect4: input (1,2,6,7) -> policy(7), value(1)
    """
    def __init__(self):
        if torch is None:
            raise RuntimeError("PyTorch not installed. `pip install torch` to use the NN.")
        super().__init__()
        self.conv1 = nn.Conv2d(2, 64, kernel_size=3, padding=1)  # (B,64,6,7)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.policy_head = nn.Sequential(
            nn.Conv2d(64, 2, kernel_size=1),
            nn.Flatten(),
            nn.Linear(2*6*7, 7),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Flatten(),
            nn.Linear(1*6*7, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        policy = self.policy_head(x)     # logits
        value = self.value_head(x)       # tanh in [-1,1]
        return policy, value

def load_model(path: str, device: str="cpu") -> Connect4Net:
    if torch is None:
        raise RuntimeError("PyTorch not installed.")
    model = Connect4Net().to(device)
    sd = torch.load(path, map_location=device)
    model.load_state_dict(sd)
    model.eval()
    return model
