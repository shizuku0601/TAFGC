import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression


class EDLHead(nn.Module):
    def __init__(self, dimensions, classes, hidden=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(dimensions, hidden), nn.ReLU(),
            nn.Linear(hidden, classes),
        )
        self.classes = classes

    def forward(self, features):
        alpha = 1 + F.softplus(self.network(features))
        strength = alpha.sum(dim=1, keepdim=True)
        return alpha / strength, self.classes / strength[:, 0], alpha


def evidential_loss(probability, alpha, labels, anneal):
    target = F.one_hot(labels.long(), alpha.shape[1]).to(alpha.dtype)
    strength = alpha.sum(dim=1, keepdim=True)
    fit = (target - probability).square().sum(dim=1)
    variance = (
        alpha * (strength - alpha) / (strength.square() * (strength + 1))
    ).sum(dim=1)
    adjusted = target + (1 - target) * alpha
    total = adjusted.sum(dim=1, keepdim=True)
    classes = alpha.shape[1]
    kl = (
        torch.lgamma(total) - torch.lgamma(adjusted).sum(dim=1, keepdim=True)
        - torch.lgamma(alpha.new_tensor(float(classes)))
        + ((adjusted - 1) * (
            torch.digamma(adjusted) - torch.digamma(total)
        )).sum(dim=1, keepdim=True)
    )[:, 0]
    return (fit + variance + anneal * kl).mean()


def _support(vacuity, agreement, fit, output):
    if np.unique(agreement[fit]).size < 2:
        return np.full(output.sum(), agreement[fit].mean())
    model = LogisticRegression().fit(
        -vacuity[fit, None], agreement[fit],
    )
    return model.predict_proba(-vacuity[output, None])[:, 1]


def calibrated_weights(
    probability_hsi, vacuity_hsi, probability_aux, vacuity_aux,
    pseudo_labels, coordinates, folds=3, block_size=32,
):
    probability_hsi = np.asarray(probability_hsi)
    probability_aux = np.asarray(probability_aux)
    vacuity_hsi = np.asarray(vacuity_hsi)
    vacuity_aux = np.asarray(vacuity_aux)
    pseudo_labels = np.asarray(pseudo_labels)
    coordinates = np.asarray(coordinates)
    block = coordinates // block_size
    block_id = block[:, 0] * (block[:, 1].max() + 1) + block[:, 1]
    unique = np.unique(block_id)
    fold = np.zeros(len(block_id), dtype=int)
    for index, value in enumerate(unique):
        fold[block_id == value] = index % folds

    support_hsi = np.empty(len(fold))
    support_aux = np.empty(len(fold))
    global_weight = np.empty(len(fold))
    reliability_hsi = np.empty(len(fold))
    reliability_aux = np.empty(len(fold))
    agreement_hsi = probability_hsi.argmax(axis=1) == pseudo_labels
    agreement_aux = probability_aux.argmax(axis=1) == pseudo_labels
    for index in range(folds):
        fit = fold != index
        output = fold == index
        support_hsi[output] = _support(vacuity_hsi, agreement_hsi, fit, output)
        support_aux[output] = _support(vacuity_aux, agreement_aux, fit, output)
        reliability_hsi[output] = support_hsi[output].mean()
        reliability_aux[output] = support_aux[output].mean()
        global_weight[output] = (
            reliability_hsi[output] / (
                reliability_hsi[output] + reliability_aux[output]
            )
        )

    def logit(value):
        value = np.clip(value, 1e-4, 1 - 1e-4)
        return np.log(value) - np.log1p(-value)

    local_gap = logit(support_hsi) - logit(support_aux)
    global_gap = logit(reliability_hsi) - logit(reliability_aux)
    local_weight = 1 / (1 + np.exp(-(
        logit(global_weight)
        + np.abs(support_hsi - support_aux) * (local_gap - global_gap)
    )))
    return local_weight
