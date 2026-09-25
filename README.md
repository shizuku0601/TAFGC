# TAFGC
## Method Overview

### Overall Framework

The usefulness of different modalities varies across a remote sensing scene, motivating locally adaptive fusion. Without category labels, modality-support assessment requires a shared clustering reference that accommodates within-class observation diversity. TAFGC couples adaptive fine-grained discovery with evidential learning: the clustering partition provides a common task for modality assessment, while calibrated support guides fusion and subsequent structural updates.
![方法框架](images/overview.png)

### Fine-Grained Discovery (FGD)

FGD preserves local observation variations through fine-grained components before organizing them into target categories. Components are first constructed from the fused representations. A shared grouping network then combines component features with aggregated pixel-level relations to learn component-to-category assignments under a structural loss. The resulting pixel-level partition serves as the shared pseudo-label reference. Components and their category assignments are periodically updated as the fused representations evolve.

### Evidential Learning and Trustworthy Multimodal Fusion (TMF)

Two independent evidential heads predict the shared pseudo-labels from their respective modality representations to assess modality support. To address differences in raw evidence scales, TMF calibrates evidence insufficiency into comparable probabilities of agreement with the reference partition. These scores guide local fusion through a fold-wide baseline and pixel-specific adjustments. The updated fusion weights influence subsequent representation learning and fine-grained discovery, linking modality adaptation with structural refinement.
