import numpy as np
import torch
import os

ZERO = 1e-8


def pytorch_cos_sim(a: torch.Tensor, b: torch.Tensor):
    """
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
    This function can be used as a faster replacement for 1-scipy.spatial.distance.cdist(a,b)
    :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = a / (a.norm(dim=1)[:, None] + 1e-10)
    b_norm = b / (b.norm(dim=1)[:, None] + 1e-10)
    return torch.mm(a_norm, b_norm.transpose(0, 1))


def normalized_correlation_matrix(nodes):
    # Nodes: (sz, 768)
    A = pytorch_cos_sim(nodes, nodes)
    A = (A + 1) / 2

    D = torch.diag((torch.sum(A, dim=1) + len(nodes)) ** (-0.5))
    D.masked_fill_(D == float('inf'), 0.)
    A = D @ A @ D

    return A