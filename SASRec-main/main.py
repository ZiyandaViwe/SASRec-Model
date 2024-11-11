import argparse
import tensorflow.compat.v1 as tf
from data.DataProcessing import process_data
from util import data_partition
from sampler import WarpSampler
from model import Model
from tqdm import tqdm
from util import *
import time

tf.compat.v1.disable_eager_execution()

def main(args):
    user_data, num_users, num_items = process_data(args.data_path)
    print(f"Loaded {num_users} users and {num_items} items.") 
    train_data, valid_data, test_data, usernum, itemnum = data_partition(user_data)

    cc = 0.0
    for u in train_data:
        cc += len(train_data[u])
    print('average sequence length: %.2f' % (cc / len(train_data)))

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)

    # Initialize model, sampler, and train model with data
    model = Model(usernum, itemnum, args)
    sampler = WarpSampler(train_data, usernum, args.batch_size)
    sess.run(tf.global_variables_initializer())

    # Define number of batches
    num_batch = len(train_data) // args.batch_size

    try:
        T = 0.0
        t0 = time.time()

        for epoch in range(1, args.num_epochs + 1):
            for step in tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
                u, seq, pos, neg = sampler.next_batch()
                auc, loss, _ = sess.run([model.auc, model.loss, model.train_op],
                                         {model.u: u, model.input_seq: seq, model.pos: pos, model.neg: neg,
                                          model.is_training: True})

            if epoch % 20 == 0:
                t1 = time.time() - t0
                T += t1
                print('Evaluating')
                t_test = evaluate(model, test_data, args, sess)
                t_valid = evaluate_valid(model, valid_data, args, sess)
                print()
                print('epoch:%d, time: %f(s), valid (NDCG@10: %.4f, HR@10: %.4f), test (NDCG@10: %.4f, HR@10: %.4f)' % (
                    epoch, T, t_valid[0], t_valid[1], t_test[0], t_test[1]))

                # f.write(str(t_valid) + ' ' + str(t_test) + '\n')
                # f.flush()
                t0 = time.time()

    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        sampler.close()

    print("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a recommendation model.")

    parser.add_argument('--data_path', type=str, default='data/testdata.csv', help="Path to the dataset file.")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size for training.")
    parser.add_argument('--num_epochs', type=int, default=201, help="Number of training epochs.")
    parser.add_argument('--maxlen', type=int, default=50, help="Maximum sequence length.")
    parser.add_argument('--dataset', type=str, default='testdata', help="Dataset name.")
    parser.add_argument('--train_dir', type=str, default='/path/to/train_dir', help="Directory to save training outputs.")
    parser.add_argument('--hidden_units', type=int, default=48, help='Number of hidden units in the model')
    parser.add_argument('--l2_emb', type=float, default=0.0, help='L2 regularization strength for embeddings')
    parser.add_argument('--dropout_rate', type=float, default=0.5, help='Dropout rate for the model')
    parser.add_argument('--num_blocks', type=int, default=2, help='Number of blocks in the model')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for the Adam optimizer')

    args = parser.parse_args()
    main(args)
