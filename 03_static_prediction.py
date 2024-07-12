import os
import glob
import pickle
import argparse
import numpy as np

from utilities import GNNPredictor

target = {'setcover': 'intgap_nonorm',
          'cauctions': 'ratio',
          'gisp': 'intgap_nonorm',
          'mixed': 'intgap_nonorm'}

samples_dirs = {'setcover': 'data/samples/setcover/750_1000_0.05',
                'cauctions': 'data/samples/cauctions/200_1000',
                'gisp': 'data/samples/GISP/80_6',
                'mixed': None}


''' This script evaluates the performance of three predictors:
        - estim: Predictor based on the estimated bound of the open nodes.
        - rank-1: predictor based on the set of rank-1 open nodes.
        - gnn: predictor based on the trained gnn model based on root features.
'''

def load_data(sample_files, gnn_predictor, eps_values):

    predictor_estim = []
    predictor_rank1 = []
    predictor_gnn = {eps: [] for eps in eps_values}
    gap = []
    y = []

    for sample_file in sample_files:
        with open(sample_file, 'rb') as f:
            sample = pickle.load(f)

        target = sample["target"]
        y.append(target)

        predictor_estim.append( sample["predictor_estim"] )
        predictor_rank1.append( sample["predictor_rank1"] )
        for eps in eps_values:
            p = gnn_predictor.predict_bool(sample, eps)
            predictor_gnn[eps].append(p)

        gap.append(100 * sample["incumbent"] / sample["optobj"])

    predictor_estim = np.array(predictor_estim, dtype=np.bool_)
    predictor_rank1 = np.array(predictor_rank1, dtype=np.bool_)
    for eps in eps_values:
        predictor_gnn[eps] = np.array(predictor_gnn[eps], dtype=np.bool_)
    gap = np.array(gap, dtype=np.float32)
    y = np.array(y, dtype=np.bool_)
    
    predictions = (predictor_estim, predictor_rank1, predictor_gnn)
    return predictions, gap, y


def evaluate(prediction, y, gap):

    # accuracy #
    accuracy = np.mean((prediction == y))
    print(f'Accuracy is {accuracy}')

    # stats on false pos/negs #
    ntest = y.shape[0]
    num_fp = np.sum(np.logical_and(prediction == 1.0, y == 0.0))
    num_fn = np.sum(np.logical_and(prediction == 0.0, y == 1.0))
    print(f'% false positives is {num_fp/ntest}')
    print(f'% false negatives is {num_fn/ntest}')

    # stats on false positives #
    fp = np.logical_and(prediction == 1.0, y == 0.0)
    fp_gap = [gap[i] for i in range(ntest) if fp[i]]
    print(f'average inc/opt on false positives: {np.mean(fp_gap)/100}')

    return accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'problem',
        help='MILP instance type to process.',
        choices=['setcover', 'gisp', 'mixed', 'cauctions'],
    )
    parser.add_argument(
        '-s', '--seed',
        help='Random generator seed (default 0).',
        type=int,
        default=17,
    )
    args = parser.parse_args()
    rng = np.random.RandomState(args.seed)

    # get the samples #
    if args.problem == 'mixed':
        valid_samples = []
        test_samples = []
        for prob in ['cauctions', 'gisp', 'setcover']:
            samples_dir = samples_dirs[prob]
            valid_samples += glob.glob(f'{samples_dir}/valid/*.pkl')
            test_samples += glob.glob(f'{samples_dir}/test/*.pkl')
        valid_samples = rng.choice(valid_samples, 2000, replace=False)
        test_samples = rng.choice(test_samples, 2000, replace=False)
    else:
        samples_dir = samples_dirs[args.problem]
        valid_samples = glob.glob(f'{samples_dir}/valid/*.pkl')
        test_samples = glob.glob(f'{samples_dir}/test/*.pkl')

    eps_values = [-0.02, -0.015, -0.01, 0.0, 0.01, 0.015, 0.02]

    # load gnn predictor #
    gnn_target = target[args.problem]
    params = f'trained_models/{args.problem}/{gnn_target}/train_params.pkl'
    gnn_predictor = GNNPredictor(params, gnn_target)


    # load validation data #
    predictions, gap, y = load_data(valid_samples, gnn_predictor, eps_values)
    predictor_estim, predictor_rank1, predictor_gnn = predictions
    print('Average target on test: ', np.mean(y))


    # grid search over epsilon values #
    best_eps = None
    best_accuracy = 0.0
    for eps in eps_values:
        print(f'\n~~ Predictor based on gnn, eps {eps} ~~')
        accuracy = evaluate(predictor_gnn[eps], y, gap)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_eps = eps


    # load test data #
    predictions, gap, y = load_data(test_samples, gnn_predictor, [best_eps])
    predictor_estim, predictor_rank1, predictor_gnn = predictions
    print('Average target on test: ', np.mean(y))


    # final evaluation #
    print('\n\n~~ Predictor based on estimate ~~')
    evaluate(predictor_estim, y, gap)
    print('\n~~ Predictor based on rank1 ~~')
    evaluate(predictor_rank1, y, gap)
    print(f'\n~~ Predictor based on gnn, eps {best_eps} ~~')
    evaluate(predictor_gnn[best_eps], y, gap)
    
    



    



