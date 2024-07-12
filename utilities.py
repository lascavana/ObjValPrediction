import pickle
import datetime
import numpy as np

import torch
import torch_geometric
import torch.nn.functional as F

from models import GNNPolicy


def log(str, logfile=None):
  str = f'[{datetime.datetime.now()}] {str}'
  print(str)
  if logfile is not None:
    with open(logfile, mode='a') as f:
      print(str, file=f)


def pad_tensor(input_, pad_sizes, pad_value=-1e8):
  max_pad_size = pad_sizes.max()
  output = input_.split(pad_sizes.cpu().numpy().tolist())
  output = torch.stack([F.pad(slice_, (0, max_pad_size-slice_.size(0)), 'constant', pad_value)
                        for slice_ in output], dim=0)
  return output

def numerics(value):
    if value < 1e-6 and value > 0.0:
        return 1e-6
    if value < 0.0 and value > 1e-6:
        return -1e-6
    return value

class BipartiteNodeData(torch_geometric.data.Data):
    def __init__(self, constraint_features, edge_index, edge_features, variable_features,
                 target, root_lp_obj, num_c_nodes, num_v_nodes):
        super().__init__()
        self.constraint_features = constraint_features
        self.edge_index = edge_index
        self.edge_features = edge_features
        self.variable_features = variable_features
        self.target = target
        self.root_lp_obj = root_lp_obj
        self.num_c_nodes = num_c_nodes
        self.num_v_nodes = num_v_nodes

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index':
            return torch.tensor([[self.constraint_features.size(0)], [self.variable_features.size(0)]])
        else:
            return super().__inc__(key, value, *args, **kwargs)


class GraphDataset(torch_geometric.data.Dataset):
    def __init__(self, sample_files, target='intgap'):
        super().__init__(root=None, transform=None, pre_transform=None)
        self.sample_files = sample_files
        self.target = target

    def len(self):
        return len(self.sample_files)

    def get(self, index):
        # read sample #
        with open(self.sample_files[index], 'rb') as f:
            sample = pickle.load(f)
        constraint_features, (edge_indices, edge_features), variable_features = sample['obs']
        root_lp_obj = sample['root_lp_obj']

        # get target #
        if self.target == 'optobj':
            target = sample['optobj']
        elif self.target == 'ratio':
            target = sample['optobj'] / numerics(sample['root_lp_obj'])
        elif self.target == 'intgap':
            target = 100 * (sample['optobj'] - sample['root_lp_obj']) / abs(numerics(sample['root_lp_obj']))
        elif self.target == 'intgap_nonorm':
            target = sample['optobj'] - sample['root_lp_obj']
        else:
            raise NotImplementedError

        # transform to tensor #
        constraint_features = torch.FloatTensor(constraint_features)
        edge_indices = torch.LongTensor(edge_indices.astype(np.int32))
        edge_features = torch.FloatTensor(np.expand_dims(edge_features, axis=-1))
        variable_features = torch.FloatTensor(variable_features)
        variable_features = torch.cat((variable_features[:,:13], variable_features[:,15:]), dim=-1)
        target = torch.FloatTensor([target])
        num_c_nodes = torch.IntTensor([constraint_features.shape[0]])
        num_v_nodes = torch.IntTensor([variable_features.shape[0]])

        graph = BipartiteNodeData(constraint_features, edge_indices, edge_features, 
                                  variable_features, target, root_lp_obj, num_c_nodes, num_v_nodes)        

        return graph



class Scheduler(torch.optim.lr_scheduler.ReduceLROnPlateau):
    def __init__(self, optimizer, **kwargs):
        super().__init__(optimizer, **kwargs)

    def step(self, metrics):
        # convert `metrics` to float, in case it's a zero-dim Tensor
        current = float(metrics)
        self.last_epoch =+1

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs == self.patience:
            self._reduce_lr(self.last_epoch)

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]


class GNNPredictor:
    def __init__(self, params, target, eps=0.02):
        self.eps = eps
        self.target = target
        self.policy = GNNPolicy()
        self.policy.load_state_dict( torch.load(params,
                                                map_location=torch.device('cpu')) )

    def predict(self, sample):
        constraint_features, (edge_indices, edge_features), variable_features = sample['obs']
        root_lp_obj = sample["root_lp_obj"]

        # transform to tensor #
        constraint_features = torch.FloatTensor(constraint_features)
        edge_indices = torch.LongTensor(edge_indices.astype(np.int32))
        edge_features = torch.FloatTensor(np.expand_dims(edge_features, axis=-1))
        variable_features = torch.FloatTensor(variable_features)
        variable_features = torch.cat((variable_features[:,:13], variable_features[:,15:]), dim=-1)
        num_c_nodes = torch.IntTensor([constraint_features.shape[0]])
        num_v_nodes = torch.IntTensor([variable_features.shape[0]])

        graph = BipartiteNodeData(constraint_features, edge_indices, edge_features, 
                                variable_features, None, None, num_c_nodes, num_v_nodes)

        # run graph through GNN #
        with torch.no_grad():
            output = self.policy(graph).numpy()[0]
        
        # transform into objective value #
        if self.target == 'optobj':
            y = output
        elif self.target == 'ratio':
            y = output * numerics(root_lp_obj)
        elif self.target == 'intgap':
            y = output * abs(numerics(root_lp_obj))/100 + root_lp_obj
        elif self.target == 'intgap_nonorm':
            y = output + root_lp_obj
        else:
            raise NotImplementedError
        
        return y

    def predict_bool(self, sample, eps=None):
        if eps is None:
            eps = self.eps 

        incumbent = sample["incumbent"]
        prediction = self.predict(sample)

        if incumbent < prediction + abs(prediction) * eps:
            return 1
        else:
            return 0
