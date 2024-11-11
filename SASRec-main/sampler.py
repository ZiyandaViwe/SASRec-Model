import numpy as np
import random
from multiprocessing import Process, Queue

def sample_function(user_train, usernum, batch_size, result_queue, SEED):
    def sample():
        user = np.random.randint(1, usernum + 1)
        print(f"Sampling user {user}")  # Debug message to track the progress

        
        # Ensure the user has enough interactions (more than 1)
        while user not in user_train or len(user_train[user]) <= 1:
            user = np.random.randint(1, usernum + 1)  # Pick a new user if not enough data
        
        # Initialize the sequences
        seq, pos, neg = np.zeros([10], dtype=np.int32), np.zeros([10], dtype=np.int32), np.zeros([10], dtype=np.int32)
        nxt, idx = user_train[user][-1], 9  # Start with the last interaction
        ts = set(user_train[user])  # Set of interactions to avoid repeating

        for i in reversed(user_train[user][:-1]):  # Iterate over the user's history (except the last interaction)
            seq[idx], pos[idx] = i, nxt  # Store the sequence and the positive sample (next item)
            neg[idx] = random.randint(1, usernum) if nxt != 0 else 0  # Random negative sample, avoid 0 if possible
            nxt, idx = i, idx - 1  # Update the next item and the index
            if idx == -1:  # Stop when the sequence is full
                break

        print(f"Generated batch for user {user}")  # Debug message


        return (user, seq, pos, neg)

    np.random.seed(SEED)  # Set the random seed for reproducibility
    while True:
        result_queue.put(list(zip(*[sample() for _ in range(batch_size)])))  # Put the batch of samples into the result queue

class WarpSampler(object):
    def __init__(self, User, usernum, batch_size=64, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = [
            Process(target=sample_function, args=(User, usernum, batch_size, self.result_queue, np.random.randint(int(2e9))))
            for _ in range(n_workers)
        ]
        for p in self.processors:
            p.daemon = True
            p.start()

    def next_batch(self):
        return self.result_queue.get()  # Return the next batch of samples

    def close(self):
        for p in self.processors:
            p.terminate()  # Terminate the worker processes
            p.join()  # Ensure the processes are properly joined after termination
