# Curriculum-Meta-based-Pre-training

Description: Implementation of Curriculum Meta-based Pre-training with Synthetic Data

## Abstract
The extensive data requirements of neural networks have sparked questions about learning from small datasets, generalizing to unseen domains, and maintaining robustness against input perturbations. Leveraging pre-trained models from large datasets like ImageNet is a common strategy to address these issues, but this approach has its drawbacks. These publicly available datasets impose strict design constraints, such as input size and model architecture, along with concerns related to privacy, usage rights, and ethical issues. Synthetic data offer an attractive alternative, addressing these challenges by allowing greater control over training data and enabling moderation to ensure diversity, fairness, and privacy. This study presents a novel framework, Curriculum Meta-based Pre-training (CMP), which proposes to pre-train models through a sequence of increasingly challenging pretext tasks, aiming at progressively introducing prior knowledge into the models from general and abstract to target specific. Positioned at the intersection of mature fields known for their effectiveness in learning general representations, this domain- and model-agnostic framework represents a novel approach in the realm of pre-training methodologies. This work evaluates the CMP ability to leverage synthetic shapes and patterns to enhance the performance of three different convolutional neural networks (CNNs) across five popular real-image datasets. Despite using significantly less data, CMP-trained models consistently exhibit enhanced generalization performance and convergence speed across all CNNs and datasets, laying the foundations for this promising line of research.

## Dataflow

![dataflow](https://github.com/LMAPcoder/Curriculum-Meta-based-Pre-training/blob/main/dataflow.jpg)
