import os
import pickle 
import pathlib
import argparse
import numpy as np

from datetime import datetime


problem_folders = {
        'setcover': 'setcover/750_1000_0.05',
        'facility': 'facility/100_100_5',
        'gisp': 'GISP/80_6',
        'cauctions': 'cauctions/200_1000'
    }

def pretrain(policy, pretrain_loader):
    policy.pre_train_init()
    for batch in pretrain_loader:
        batch.to(device)
        policy.pre_train(batch)
    policy.pre_train_end()


def process(policy, data_loader, optimizer=None):
    mean_loss = 0
    n_samples_processed = 0
    with torch.set_grad_enabled(optimizer is not None):
        for batch in data_loader:
            batch = batch.to(device)
            output = policy(batch)
            loss = F.mse_loss(output, batch.target, reduction='mean')

            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            mean_loss += loss.item() * batch.num_graphs
            n_samples_processed += batch.num_graphs

    mean_loss /= n_samples_processed
    return mean_loss

def transform(x, root_lp_obj, target):
    if target == 'optobj':
        y = x
    elif target == 'ratio':
        y = x * numerics(root_lp_obj)
    elif target == 'intgap':
        y = x * abs(numerics(root_lp_obj))/100 + root_lp_obj
    elif target == 'intgap_nonorm':
        y = x + root_lp_obj
    else:
        raise NotImplementedError
    return y

def test(policy, data_loader, target):
    true_val = []
    pred_val = []
    with torch.no_grad():
        for sample in data_loader:
            sample = sample.to(device)
            output = policy(sample)

            pred = transform(output, sample.root_lp_obj, target)
            true = transform(sample.target, sample.root_lp_obj, target)

            true_val.append(true.item())
            pred_val.append(pred.item())

    true_val = np.array(true_val)
    pred_val = np.array(pred_val)

    err = 100 * np.mean(np.abs(true_val - pred_val) / true_val)
    return err


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'problem',
        help='MILP instance type to process.',
        choices=['setcover', 'gisp', 'facility', 'cauctions', 'mixed'],
    )
    parser.add_argument(
        '-s', '--seed',
        help='Random generator seed (default 0).',
        type=int,
        default=17,
    )
    parser.add_argument(
        '-t', '--target',
        help='Learning target.',
        choices=['optobj', 'ratio', 'intgap', 'intgap_nonorm'],
        default='ratio',
    )
    parser.add_argument(
        '-g', '--gpu',
        help='CUDA GPU id (-1 for CPU).',
        type=int,
        default=0,
    )
    args = parser.parse_args()
    seed = args.seed


    # hyper parameters #
    max_epochs = 1000
    batch_size = 32
    pretrain_batch_size = 128
    valid_batch_size = 128
    train_size = 10000
    valid_size = 2000
    test_size = 2000
    lr = 1e-3
    target = args.target


    # directories #
    timestamp = datetime.now().strftime('%Y-%m-%d') + "--" + datetime.now().strftime('%H.%M.%S')
    running_dir = f"trained_models/{args.problem}/{timestamp}"
    os.makedirs(running_dir, exist_ok=True)


    # pytorch setup #
    if args.gpu == -1:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        device = "cpu"
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = f'{args.gpu}'
        device = f"cuda:0"
    import torch
    import torch_geometric
    import torch.nn.functional as F
    from models import GNNPolicy
    from utilities import log, GraphDataset, Scheduler, numerics


    # seeding #
    rng = np.random.RandomState(args.seed)
    torch.manual_seed(args.seed)


    # log #
    logfile = os.path.join(running_dir, 'train_log.txt')
    if os.path.exists(logfile):
        os.remove(logfile)
    log(f"max_epochs: {max_epochs}", logfile)
    log(f"batch_size: {batch_size}", logfile)
    log(f"pretrain_batch_size: {pretrain_batch_size}", logfile)
    log(f"valid_batch_size : {valid_batch_size }", logfile)
    log(f"lr: {lr}", logfile)
    log(f"problem: {args.problem}", logfile)
    log(f"gpu: {args.gpu}", logfile)
    log(f"seed {args.seed}", logfile)
    log(f'target: {target}', logfile)


    # get samples #
    if args.problem == 'mixed':
        train_files, valid_files, test_files = [], [], []
        for problem in ['cauctions', 'gisp', 'setcover']:
            problem_dir = problem_folders[problem] 
            train_files += [str(file) for file in (pathlib.Path(f'data/samples')/problem_dir/'train').glob('sample_*.pkl')]
            valid_files += [str(file) for file in (pathlib.Path(f'data/samples')/problem_dir/'valid').glob('sample_*.pkl')]
            test_files += [str(file) for file in (pathlib.Path(f'data/samples')/problem_dir/'test').glob('sample_*.pkl')]
        train_files = rng.choice(train_files, train_size, replace=False)
        valid_files = rng.choice(valid_files, valid_size, replace=False)
        test_files = rng.choice(test_files, test_size, replace=False)
        pretrain_files = [f for i, f in enumerate(train_files) if i % 10 == 0]
    else:
        problem_dir = problem_folders[args.problem]
        train_files = [str(file) for file in (pathlib.Path(f'data/samples')/problem_dir/'train').glob('sample_*.pkl')]
        valid_files = [str(file) for file in (pathlib.Path(f'data/samples')/problem_dir/'valid').glob('sample_*.pkl')]
        test_files = [str(file) for file in (pathlib.Path(f'data/samples')/problem_dir/'test').glob('sample_*.pkl')]
        pretrain_files = [f for i, f in enumerate(train_files) if i % 10 == 0]

    # create data loaders #
    pretrain_data = GraphDataset(pretrain_files, target)
    pretrain_loader = torch_geometric.loader.DataLoader(pretrain_data, pretrain_batch_size, shuffle=False)
    valid_data = GraphDataset(valid_files, target)
    valid_loader = torch_geometric.loader.DataLoader(valid_data, valid_batch_size, shuffle=False)
    test_data = GraphDataset(test_files, target)
    test_loader = torch_geometric.loader.DataLoader(test_data, 1, shuffle=False)
    
    # create policy #
    policy = GNNPolicy().to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr) 
    scheduler = Scheduler(optimizer, mode='min', patience=10, factor=0.2)

    # pretrain #
    pretrain(policy, pretrain_loader)
    
    # training loop #
    for epoch in range(max_epochs + 1):
        log(f"EPOCH {epoch}...", logfile)

        # load data #
        epoch_train_files = rng.choice(train_files, int(np.floor(train_size/batch_size))*batch_size, replace=True)
        train_data = GraphDataset(epoch_train_files, target)
        train_loader = torch_geometric.loader.DataLoader(train_data, batch_size, shuffle=True)
        
        # train #
        train_loss = process(policy, train_loader, optimizer)
        log(f"TRAIN LOSS: {train_loss:0.3f} ", logfile)

        # validation #
        valid_loss = process(policy, valid_loader, None)
        log(f"VALID LOSS: {valid_loss:0.3f} ", logfile)

        scheduler.step(valid_loss)
        if scheduler.num_bad_epochs == 0:
            torch.save(policy.state_dict(), pathlib.Path(running_dir)/'train_params.pkl')
            log(f"  best model so far", logfile)
        elif scheduler.num_bad_epochs == 10:
            log(f"  10 epochs without improvement, decreasing learning rate", logfile)
        elif scheduler.num_bad_epochs == 20:
            log(f"  20 epochs without improvement, early stopping", logfile)
            break
    
    # final validation #
    valid_loader = torch_geometric.loader.DataLoader(valid_data, 1, shuffle=False)
    err = test(policy, valid_loader, target)
    log(f"  Valid error {err:0.3f}", logfile)

    # final test #
    err = test(policy, test_loader, target)
    log(f"  Test error {err:0.3f}", logfile)

    # extra tests for the mixed case #
    if args.problem == 'mixed':
        for problem in ['cauctions', 'gisp', 'setcover']:
            problem_dir = problem_folders[problem] 
            test_files = [str(file) for file in (pathlib.Path(f'data/samples')/problem_dir/'test').glob('sample_*.pkl')]
            test_data = GraphDataset(test_files, target)
            test_loader = torch_geometric.loader.DataLoader(test_data, 1, shuffle=False)
            err = test(policy, test_loader, target)
            log(f"  Test error on {problem} is {err:0.3f}", logfile)

    
