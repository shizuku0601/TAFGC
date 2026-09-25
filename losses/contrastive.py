import torch
import torch.nn.functional as F


def relation_ranking_loss(embeddings, positive_table, hsi_descriptors, temperature):
    embeddings = F.normalize(embeddings, dim=1)
    similarity = embeddings @ embeddings.T
    reference = torch.arange(len(embeddings), device=embeddings.device)
    positive = positive_table[:, 0].to(embeddings.device)
    valid = positive >= 0
    positive_score = similarity[reference, positive.clamp_min(0)]

    hsi_descriptors = F.normalize(hsi_descriptors.to(embeddings.device), dim=1)
    hsi_similarity = hsi_descriptors @ hsi_descriptors.T
    negative = hsi_similarity < 0.9
    negative[reference, reference] = False
    negative[reference[valid], positive[valid]] = False

    penalty = F.softplus(
        (similarity - positive_score[:, None]) / temperature,
    )
    per_pixel = (penalty * negative).sum(dim=1) / negative.sum(dim=1).clamp_min(1)
    return per_pixel[valid & negative.any(dim=1)].mean()
