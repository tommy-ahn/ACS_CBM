import torch
import torch.nn as nn


class Node(nn.Module):
    def __init__(self):
        pass


class Leaf(nn.Module):
    def __init__(self):
        pass


class NeuralConceptTree(nn.Module):
    def __init__(self, dim, n_tasks, depth) -> None:
        super().__init__()
        self.dim = dim
        self.n_tasks = n_tasks
        self.depth = depth

    def forward(self, x):
        pass
