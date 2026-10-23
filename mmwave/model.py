"""PointNet-style encoder + multitask head (class, position, heading).

Plain Linear/ReLU layers only, so the browser kernels can mirror it exactly.
"""
import mlx.core as mx
import mlx.nn as nn

ENC = [6, 64, 128, 256]
HEAD = [256, 128, 64, 8]  # 4 class logits, (x, y), (sin yaw, cos yaw)


class PointNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = [nn.Linear(a, b) for a, b in zip(ENC[:-1], ENC[1:])]
        self.head = [nn.Linear(a, b) for a, b in zip(HEAD[:-1], HEAD[1:])]
        self.drop = nn.Dropout(0.2)

    def __call__(self, x, mask):
        h = x
        for layer in self.enc:
            h = nn.relu(layer(h))
        # ReLU outputs are >= 0, so zeroing padded points makes them inert under max.
        h = (h * mask[..., None]).max(axis=1)
        for layer in self.head[:-1]:
            h = self.drop(nn.relu(layer(h)))
        return self.head[-1](h)
