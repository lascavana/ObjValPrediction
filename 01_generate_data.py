import os
import glob
import ecole
import pickle
import argparse
import pyscipopt as scip

from plugins import EventHandler, RootBoundReward

def get_root_state(instance, seed):
    scip_params = {'presolving/maxrestarts': 0,
                   'separating/maxroundsroot': 0,
                   'reoptimization/strongbranchinginit': 0}

    # get root node observation #
    env = ecole.environment.Branching(
        scip_params=scip_params,
        observation_function=ecole.observation.NodeBipartite(),
        reward_function=RootBoundReward(),
    )
    env.seed(seed)
    obs, _, root_lp_obj, done, _ = env.reset(instance)
    if done:
        raise Exception("Instance solved at the root node")

    obs= (obs.row_features,
         (obs.edge_features.indices,
          obs.edge_features.values),
          obs.column_features)

    return obs, root_lp_obj


def generate_samples(instances, out_dir, dataset_size, seed=0):

    # init pyscipopt model #
    m = scip.Model()
    m.setIntParam('display/verblevel', 0)
    event_handler = EventHandler(seed=seed, sample_prob=0.02)
    m.includeEventhdlr(event_handler, "CustomEventHandler", "")

    sample_count = 0
    for instance in instances:
        print(f'Processing instance {instance}')

        # get root node state #
        obs, root_lp_obj = get_root_state(instance, seed)

        # solve #
        m.readProblem(instance)
        m.optimize()
        optobj = m.getUpperbound()

        # post-process and save data #
        samples = event_handler.samples
        for j, sample in enumerate(samples):
            sample["obs"] = obs
            sample["root_lp_obj"] = root_lp_obj

            target = (sample["incumbent"] == optobj)
            sample["target"] = target
            sample["optobj"] = optobj
            sample["incumbent"] = sample["incumbent"]

            with open(f'{out_dir}/sample_{sample_count+j}.pkl', 'wb') as f:
                pickle.dump(sample, f)
        
        sample_count += len(samples)

        m.freeProb()
        event_handler.reset()

        print(f'Number of samples collected so far is {sample_count}')

        if sample_count > dataset_size: break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'problem',
        help='MILP instance type to process.',
        choices=['setcover', 'gisp', 'facility', 'cauctions'],
    )
    parser.add_argument(
        '-s', '--seed',
        help='Random generator seed (default 0).',
        type=int,
        default=17,
    )
    args = parser.parse_args()
    seed = args.seed

    train_size = 10000
    valid_size = 2000
    test_size = 2000

    if args.problem == 'setcover':
        train_dir = 'data/instances/setcover/train_750_1000_0.05'
        valid_dir = 'data/instances/setcover/valid_750_1000_0.05'
        test_dir = 'data/instances/setcover/test_750_1000_0.05'
        out_dir = 'data/samples/setcover/750_1000_0.05'
    elif args.problem == 'gisp':
        train_dir = 'data/instances/GISP/train_80_6'
        valid_dir = 'data/instances/GISP/valid_80_6'
        test_dir = 'data/instances/GISP/test_80_6'
        out_dir = 'data/samples/GISP/80_6'
    elif args.problem == 'facility':
        train_dir = 'data/instances/facility/train_100_100_5'
        valid_dir = 'data/instances/facility/valid_100_100_5'
        test_dir = 'data/instances/facility/test_100_100_5'
        out_dir = 'data/samples/facility/100_100_5'
    elif args.problem == 'cauctions':
        train_dir = 'data/instances/cauctions/train_200_1000'
        valid_dir = 'data/instances/cauctions/valid_200_1000'
        test_dir = 'data/instances/cauctions/test_200_1000'
        out_dir = 'data/samples/cauctions/200_1000'
    else:
        raise NotImplementedError

    os.makedirs(out_dir, exist_ok=True)

    # train #
    samples_dir = out_dir + '/train'
    os.makedirs(samples_dir, exist_ok=True)
    instances = glob.glob(f'{train_dir}/*.lp')
    generate_samples(instances, samples_dir, train_size, seed=args.seed)

    # valid #
    samples_dir = out_dir + '/valid'
    os.makedirs(samples_dir, exist_ok=True)
    instances = glob.glob(f'{valid_dir}/*.lp')
    generate_samples(instances, samples_dir, valid_size, seed=args.seed)

    # test #
    samples_dir = out_dir + '/test'
    os.makedirs(samples_dir, exist_ok=True)
    instances = glob.glob(f'{test_dir}/*.lp')
    generate_samples(instances, samples_dir, test_size, seed=args.seed)

    

    

