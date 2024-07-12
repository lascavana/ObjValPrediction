import os
import glob
import torch
import pickle
import argparse
import numpy as np

from utilities import GNNPredictor

from datetime import datetime
from sklearn import svm, preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

keys = ["gap", "tree_weight", "onodes", "mgap", "profile", "leaf_freq"]

topkeys = ["gap", "tree_weight", "mgap", "onodes", "leaf_freq"]

target = {'setcover': 'intgap_nonorm',
          'cauctions': 'ratio',
          'gisp': 'intgap_nonorm',
          'mixed': 'intgap_nonorm'}

samples_dirs = {'setcover': 'data/samples/setcover/750_1000_0.05',
                'cauctions': 'data/samples/cauctions/200_1000',
                'gisp': 'data/samples/GISP/80_6',
                'mixed': None}

def load_data_per_metric(sample_files, rng):
    X = {metric: [] for metric in keys}
    y = []

    for sample_file in sample_files:
        with open(sample_file, 'rb') as f:
            sample = pickle.load(f)

        target = sample["target"]
        y.append(target)

        for metric in keys:
            features = sample[metric]
            if metric == "onodes":
                features = features[1:]
            X[metric].append(features)  

    y = np.array(y, dtype=np.bool_)
    shuffled_indices = np.arange(y.shape[0])
    rng.shuffle(shuffled_indices)
    y = y[shuffled_indices]

    for metric in keys:
        X[metric] = np.array(X[metric], dtype=np.float32)
        X[metric] = X[metric][shuffled_indices]

    return X, y

def load_data(sample_files, rng, gnn_policy, selectedkeys=topkeys):
    X, y1, y2 = [], [], []

    for sample_file in sample_files:
        with open(sample_file, 'rb') as f:
            sample = pickle.load(f)
        features = []

        # load dynamic features #
        for metric in selectedkeys:
            if metric == "onodes":
               features.append( sample[metric][1] )
            else:
                features.append( sample[metric][0] )

        # get root features #
        optobj_prediction = gnn_policy.predict(sample)
        features.append( sample["incumbent"] / optobj_prediction)

        X.append(features)

        # load targets #
        is_optimal = sample["target"]
        ratio = sample["incumbent"] / sample["optobj"]
        y1.append(is_optimal)
        y2.append(ratio)

    X = np.array(X, dtype=np.float32)
    y1 = np.array(y1, dtype=np.bool_)
    y2 = np.array(y2, dtype=np.float32)

    shuffled_indices = np.arange(y1.shape[0])
    rng.shuffle(shuffled_indices)
    X = X[shuffled_indices]
    y1 = y1[shuffled_indices]
    y2 = y2[shuffled_indices]

    return X, y1, y2

def balance_data(X, y):
    
    N = y.shape[0]
    avg_score = np.mean(y)
    
    if avg_score > 0.55:
        minority_class = False
        n_samples_minority_class = int((1-avg_score) * N)
    elif avg_score < 0.45:
        minority_class = True
        n_samples_minority_class = int(avg_score * N)
    else:
        return X, y 

    count = 0
    idx_keep = []
    for i, sample in enumerate(y):
        if sample == minority_class:
            idx_keep.append(i)
        else:
            if count < n_samples_minority_class:
                idx_keep.append(i)
                count+=1

    idx_keep = np.array(idx_keep, dtype=np.int32)
    X = X[idx_keep]
    y = y[idx_keep]

    return X, y



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'problem',
        help='MILP instance type to process.',
        choices=['setcover', 'gisp', 'mixed', 'cauctions'],
    )
    parser.add_argument(
        '-m', '--model',
        help='ML model used for learning.',
        choices=['logistic', 'forest', 'svm', 'regression'],
        default='logistic'
    )
    parser.add_argument(
        '-s', '--seed',
        help='Random generator seed (default 0).',
        type=int,
        default=17,
    )
    args = parser.parse_args()
    rng = np.random.RandomState(seed=args.seed)
    tau = 0.99

    # get the samples #
    if args.problem == 'mixed':
        train_samples = []
        test_samples = []
        for prob in ['cauctions', 'gisp', 'setcover']:
            samples_dir = samples_dirs[prob]
            train_samples += glob.glob(f'{samples_dir}/train/*.pkl')
            test_samples += glob.glob(f'{samples_dir}/test/*.pkl')
        train_samples = rng.choice(train_samples, 10000, replace=False)
        test_samples = rng.choice(test_samples, 2000, replace=False)
    else:
        samples_dir = samples_dirs[args.problem]
        train_samples = glob.glob(f'{samples_dir}/valid/*.pkl')
        test_samples = glob.glob(f'{samples_dir}/test/*.pkl')

    # get gnn predictor #
    gnn_target = target[args.problem]
    params = f'trained_models/{args.problem}/{gnn_target}/train_params.pkl'
    gnn_policy = GNNPredictor(params, gnn_target)

    # training data #
    Xtrain, y1train, y2train = load_data(train_samples, rng, gnn_policy, topkeys)
    print('Average target on train: ', np.mean(y1train))

    # test data #
    Xtest, y1test, y2test = load_data(test_samples, rng, gnn_policy, topkeys)
    print('Average target on test: ', np.mean(y1test))

    # get normalization #
    scaler = preprocessing.StandardScaler().fit(Xtrain)
    Xtrain = scaler.transform(Xtrain)
    Xtest = scaler.transform(Xtest)

    # train model #
    if args.model == 'logistic':
        clf = LogisticRegression(random_state=0, max_iter=500).fit(Xtrain, y1train)
    elif args.model == 'forest':
        clf = RandomForestClassifier(max_depth=10, random_state=0).fit(Xtrain, y1train)
    elif args.model == 'svm':
        clf = svm.NuSVC(gamma="auto").fit(Xtrain, y1train)
    elif args.model == 'regression':
        clf = svm.SVR().fit(Xtrain, y2train)
    else:
        raise NotImplementedError

    # evaluation #
    if args.model == 'regression':
        prediction = clf.predict(Xtest)
        prediction = (prediction > tau)
        accuracy = np.mean(prediction == y1test)
    else:
        prediction = clf.predict(Xtest)
        accuracy = clf.score(Xtest, y1test)
    print(f'Accuracy is {accuracy}')

    # breakdown #
    ntest = y1test.shape[0]
    num_fp = np.sum(np.logical_and(prediction == 1.0, y1test == 0.0))
    num_fn = np.sum(np.logical_and(prediction == 0.0, y1test == 1.0))
    print(f'% false positives is {num_fp/ntest}')
    print(f'% false negatives is {num_fn/ntest}')

    # stats on false positives #
    fp = np.logical_and(prediction == 1.0, y1test == 0.0)
    fp_gap = [y2test[i] for i in range(ntest) if fp[i]]
    print(f'average inc/opt on false positives: {np.mean(fp_gap)/100}')

    print('Coefficients')
    coef = clf.coef_[0]
    print(coef)


    



