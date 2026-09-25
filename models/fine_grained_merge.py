import torch
from torch import nn

from ..utils.kmeans import spherical_kmeans


def discover_components(fused_features, classes, seed=0):
    return spherical_kmeans(fused_features.detach(), 3 * classes, seed=seed)


class SharedMergeHead(nn.Module):
    def __init__(self, feature_dim, classes, hidden_dim=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2 * feature_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, classes),
        )

    def forward(self, features, graph):
        graph = graph.to(features)
        neighbor = graph @ features / graph.sum(1, keepdim=True).clamp_min(1e-8)
        return self.network(torch.cat((features, neighbor), dim=1))

    def assignment(self, features, graph):
        return self(features, graph).softmax(1)
