import copy
import random
import numpy as np
from collections import defaultdict

def data_partition(user_data):
    user_train = {}
    user_valid = {}
    user_test = {}

    User = defaultdict(list)
    for _, row in user_data.iterrows():
        User[row['name']].append(row['no_of_ratings'])

    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = [User[user][-2]]
            user_test[user] = [User[user][-1]]

    return [user_train, user_valid, user_test, len(User), len(User[user])]

def evaluate(model, dataset, args, sess):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)
    NDCG, HT, valid_user = 0.0, 0.0, 0.0

    users = random.sample(range(1, usernum + 1), 10000) if usernum > 10000 else range(1, usernum + 1)
    
    for u in users:
        if len(train[u]) < 1 or len(test[u]) < 1:
            continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        seq[idx] = valid[u][0]
        idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1:
                break
        
        item_idx = [test[u][0]] + [random.randint(1, itemnum + 1) for _ in range(100)]
        predictions = -model.predict(sess, [u], [seq], item_idx)[0]
        rank = predictions.argsort().argsort()[0]

        valid_user += 1
        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print('.', end='', flush=True)

    return NDCG / valid_user, HT / valid_user
