
from __future__ import annotations

import torch


def structure_loss(
    assignment: torch.Tensor,
    graph: torch.Tensor,
    gamma_partition: float = 0.05,
    eps: float = 1.0e-8,
) -> dict[str, torch.Tensor]:
    graph = graph.to(device=assignment.device, dtype=assignment.dtype)
    degree = graph.sum(dim=1)
    volume = assignment.T @ degree
    classes = assignment.shape[1]
    cut = (
        assignment * (graph @ (1.0 - assignment))
    ).sum(dim=0)
    soft_cut = (cut / (volume + eps)).mean()

    normalized = assignment / torch.sqrt(volume[None, :] + eps)
    overlap = normalized.T @ (degree[:, None] * normalized)
    identity = torch.eye(
        classes, device=assignment.device, dtype=assignment.dtype,
    )
    partition = (overlap - identity).square().sum() / classes
    return {
        "total": soft_cut + float(gamma_partition) * partition,
        "soft_cut": soft_cut,
        "partition": partition,
    }
