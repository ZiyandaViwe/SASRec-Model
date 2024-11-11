## SASRec RCL (Self-Attention Sequential Recommendation with RCL)

This project implements a Sequential Recommender System based on the SASRec (Self-Attention Sequential Recommendation) model. The optimal model version integrates Ranking-based Contrastive Loss (RCL), leveraging contrastive loss principles to improve ranking and accuracy in recommendations. By encouraging the model to differentiate between similar and dissimilar items, the RCL-enhanced model exhibits improved robustness and performance over the standard SASRec approach.

# Features
Sequential Recommendation: Based on the SASRec model, leveraging self-attention to capture dependencies in sequential user-item interactions.
Ranking-based Contrastive Loss (RCL): Enhances model training by optimizing the ranking of recommended items and improving personalization.
Multi-Head Attention: Uses multi-head attention layers to model the user-item sequence effectively.
Embedding Layers: Embedding layers for user and item IDs to represent them in dense vector spaces.
Contrastive Learning Strategy: Encourages the model to focus on distinctions between similar and dissimilar items in the recommendation sequence.
Flexible Batching: WarpSampler efficiently generates batches of sequences with positive and negative samples for efficient training.
# Models
SASRec: The baseline model uses self-attention mechanisms to model sequential user patterns without contrastive loss.
SASRec with RCL: Extends SASRec with Ranking-based Contrastive Loss, encouraging the model to better distinguish relevant items, thus improving the ranking quality in recommendations.
Requirements
Python 3.x
TensorFlow 2.x
Numpy
Pandas
To run:

bash
# Requirements

- Python 3.x
- TensorFlow 2.x
- Numpy
- Pandas

To run:
bash
 python main.py --dataset testdata --train_dir /path/to/train_dir
