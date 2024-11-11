import numpy as np
import tensorflow as tf
from sampler import WarpSampler 
from data.DataProcessing import process_data  # Reusing the data processing function
from util import data_partition  #Splitting the data
from model import Model  # SASRec without RCL
import random

class RCLModel(tf.keras.Model):
    def __init__(self, usernum, itemnum, args):
        super(RCLModel, self).__init__()
        self.user_embedding = tf.keras.layers.Embedding(usernum + 1, args.hidden_units)
        self.item_embedding = tf.keras.layers.Embedding(itemnum + 1, args.hidden_units)
        self.args = args
        self.reward_threshold = 0.5  # Start with a threshold for reward-based difficulty selection

        # Define the Transformer (SASRec-style) layers
        self.transformer_blocks = [
            self.build_transformer_block() for _ in range(args.num_blocks)
        ]
        self.fc_layer = tf.keras.layers.Dense(1)  # Output layer for predicting relevance score

    def build_transformer_block(self):
        return tf.keras.Sequential([
            tf.keras.layers.MultiHeadAttention(num_heads=self.args.num_heads, key_dim=self.args.hidden_units),
            tf.keras.layers.Dropout(self.args.dropout_rate),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Dense(self.args.hidden_units, activation='relu'),
            tf.keras.layers.Dropout(self.args.dropout_rate)
        ])

    def call(self, inputs):
        user_id, item_seq, item_idx = inputs
        user_embed = self.user_embedding(user_id)
        seq_embed = self.item_embedding(item_seq)
        item_embed = self.item_embedding(item_idx)

        # Pass through transformer layers (SASRec-style)
        x = seq_embed
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)

        # Compute the logits (relevance scores) for each item in the sequence
        logits = tf.reduce_sum(x * tf.expand_dims(item_embed, axis=1), axis=-1)
        return logits

    def sample_curriculum_batch(self, train_data, difficulty='easy'):
        """
        Sample batch with controlled difficulty based on RCL principle.
        """
        batch = []
        for user, items in train_data.items():
            # Easy samples: shorter sequences, hard samples: longer sequences
            if difficulty == 'easy' and len(items) <= self.args.maxlen // 2:
                batch.append((user, items))
            elif difficulty == 'hard' and len(items) > self.args.maxlen // 2:
                batch.append((user, items))
        return random.sample(batch, min(len(batch), self.args.batch_size))

    def train_step(self, batch):
        user_ids, sequences, labels = batch
        with tf.GradientTape() as tape:
            logits = self([user_ids, sequences, labels])
            loss = tf.keras.losses.sparse_categorical_crossentropy(labels, logits)
        
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        return loss

def train_with_rcl(model, train_data, args):
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy')

    for epoch in range(args.num_epochs):
        if epoch < args.num_epochs * 0.5:
            difficulty = 'easy'  # Start with easier samples
        else:
            difficulty = 'hard'  # Progress to harder samples

        batch_data = model.sample_curriculum_batch(train_data, difficulty)
        
        user_ids = np.array([data[0] for data in batch_data])
        sequences = np.array([data[1][:-1] for data in batch_data])
        labels = np.array([data[1][-1] for data in batch_data])

        loss = model.train_step((user_ids, sequences, labels))
        print(f"Epoch {epoch + 1}, Difficulty: {difficulty}, Loss: {loss.numpy().mean()}")

        # Adapt reward threshold based on loss
        if loss.numpy().mean() < model.reward_threshold:
            model.reward_threshold *= 0.9  # Reduce threshold to increase difficulty in future batches

def main(args):
    # Step 1: Load and process the data
    user_data, num_users, num_items = process_data(args.data_path)
    train_data, valid_data, test_data, usernum, itemnum = data_partition(user_data)

    # Step 2: Initialize model and sampler
    model = RCLModel(usernum, itemnum, args)
    sampler = WarpSampler(train_data, usernum, args.batch_size)  # Using WarpSampler for batch generation
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
                  loss='sparse_categorical_crossentropy')

    # Step 3: Training loop with curriculum learning
    train_with_rcl(model, train_data, args)

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train a recommendation model with RCL.")

    parser.add_argument('--data_path', type=str, default='data/testdata.csv' ,help="Path to the dataset file.")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size for training.")
    parser.add_argument('--num_epochs', type=int, default=201, help="Number of training epochs.")
    parser.add_argument('--maxlen', type=int, default=50, help="Maximum sequence length.")
    parser.add_argument('--dataset', type=str, default='testdata', help="Dataset name.")
    parser.add_argument('--train_dir', type=str, default='/path/to/train_dir', help="Directory to save training outputs.")
    parser.add_argument('--hidden_units', type=int, default=50, help='Number of hidden units in the model')
    parser.add_argument('--l2_emb', type=float, default=0.0, help='L2 regularization strength for embeddings')
    parser.add_argument('--dropout_rate', type=float, default=0.5, help='Dropout rate for the model')
    parser.add_argument('--num_blocks', type=int, default=2, help='Number of blocks in the model')

    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for the Adam optimizer')

    args = parser.parse_args()

    main(args)
