# SASRec RCL (Self-Attention Sequential Recommendation with RCL)

This project implements a **Sequential Recommender System** based on the SASRec (Self-Attention Sequential Recommendation) model. The optimal model version integrates **Reward-Conditioned Learning (RCL)**, leveraging curriculum learning principles to train on progressively challenging data. By gradually increasing sample difficulty, the RCL-enhanced model exhibits improved robustness and performance over the standard SASRec approach.

## Features
- **Sequential Recommendation**: Based on the SASRec model, leveraging self-attention to capture dependencies in sequential user-item interactions.
- **Reward-Conditioned Learning (RCL)**: Integrates curriculum learning principles by sampling data sequences of varying difficulties.
- **Multi-Head Attention**: Uses multi-head attention layers to model the user-item sequence effectively.
- **Embedding Layers**: Embedding layers for user and item IDs to represent them in dense vector spaces.
- **Curriculum Learning Strategy**: Gradually trains on sequences from easy to hard, enhancing model capability in handling complex sequences.
- **Flexible Batching**: WarpSampler efficiently generates batches of sequences with positive and negative samples for efficient training.

## Models

1. **SASRec**: The baseline model uses self-attention mechanisms to model sequential user patterns without curriculum learning.
2. **SASRec with RCL**: Extends SASRec with Reward-Conditioned Learning, introducing curriculum learning by starting with easier samples (shorter sequences) and progressing to harder ones (longer sequences) throughout training.

## Requirements

- Python 3.x
- TensorFlow 2.x
- Numpy
- Pandas

To run:
```bash
 python main.py --dataset testdata --train_dir /path/to/train_dir










