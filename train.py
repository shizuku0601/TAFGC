import torch
import torch.nn.functional as F

from .losses.contrastive import relation_ranking_loss
from .losses.structure_loss import structure_loss
from .models.edl import calibrated_weights, evidential_loss
from .models.fine_grained_merge import discover_components


def fuse(z_hsi, z_aux, hsi_weight):
    weight = torch.as_tensor(
        hsi_weight, device=z_hsi.device, dtype=z_hsi.dtype,
    ).reshape(-1, 1)
    return torch.cat((
        weight.sqrt() * F.normalize(z_hsi, dim=1),
        (1 - weight).sqrt() * F.normalize(z_aux, dim=1),
    ), dim=1)


def refresh_structure(z_hsi, z_aux, hsi_weight, pixel_graph, merge_head, classes):
    fused = fuse(z_hsi, z_aux, hsi_weight)
    component_ids, _ = discover_components(fused, classes)
    membership = F.one_hot(component_ids, 3 * classes).to(fused.dtype)
    component_graph = membership.T @ pixel_graph @ membership
    component_features = membership.T @ fused / membership.sum(0)[:, None]
    assignment = merge_head.assignment(component_features, component_graph)
    pseudo_labels = assignment.argmax(dim=1)[component_ids]
    merge_loss = structure_loss(assignment, component_graph)["total"]
    return pseudo_labels, merge_loss


@torch.no_grad()
def update_weights(heads, z_hsi, z_aux, pseudo_labels, coordinates):
    probability_hsi, vacuity_hsi, _ = heads[0](z_hsi.detach())
    probability_aux, vacuity_aux, _ = heads[1](z_aux.detach())
    return calibrated_weights(
        probability_hsi.cpu().numpy(), vacuity_hsi.cpu().numpy(),
        probability_aux.cpu().numpy(), vacuity_aux.cpu().numpy(),
        pseudo_labels.cpu().numpy(), coordinates,
    )


def train_step(
    z_hsi, z_aux, positive_table, relation_hsi, pseudo_labels,
    edl_heads, merge_loss, optimizer, temperature, anneal,
):
    optimizer.zero_grad()
    relation = (
        relation_ranking_loss(z_hsi, positive_table, relation_hsi, temperature)
        + relation_ranking_loss(z_aux, positive_table, relation_hsi, temperature)
    )
    loss = relation
    if pseudo_labels is not None:
        for head, features in zip(edl_heads, (z_hsi, z_aux)):
            probability, _, alpha = head(features.detach())
            loss = loss + 0.5 * evidential_loss(
                probability, alpha, pseudo_labels, anneal,
            )
    if merge_loss is not None:
        loss = loss + 0.3 * merge_loss
    loss.backward()
    optimizer.step()
    return loss
