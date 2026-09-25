import torch
import torch.nn.functional as F


@torch.no_grad()
def spherical_kmeans(features, count, iterations=30, seed=0):
    features = F.normalize(features.float(), dim=1)
    generator = torch.Generator().manual_seed(seed)
    initial = torch.randperm(len(features), generator=generator)[:count]
    centers = features[initial.to(features.device)].clone()

    for _ in range(iterations):
        labels = (features @ centers.T).argmax(dim=1)
        sums = torch.zeros_like(centers).index_add_(0, labels, features)
        sizes = torch.bincount(labels, minlength=count)
        empty = sizes == 0
        if empty.any():
            replacement = torch.randint(
                len(features), (int(empty.sum()),), generator=generator,
            )
            sums[empty] = features[replacement.to(features.device)]
        centers = F.normalize(sums, dim=1)

    labels = (features @ centers.T).argmax(dim=1)
    return labels, centers
