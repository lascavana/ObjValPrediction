import torch
import torch.nn.functional as F
import torch_geometric
import numpy as np


class PreNormLayer(torch.nn.Module):
    def __init__(self, n_units, shift=True, scale=True, name=None):
        super().__init__()
        assert shift or scale
        self.register_buffer('shift', torch.zeros(n_units) if shift else None)
        self.register_buffer('scale', torch.ones(n_units) if scale else None)
        self.n_units = n_units
        self.waiting_updates = False

    def forward(self, input_):
        if self.waiting_updates:
            self.update_stats(input_)
            return input_

        if self.shift is not None:
            input_ = input_ + self.shift

        if self.scale is not None:
            input_ = input_ * self.scale

        return input_

    def start_updates(self):
        self.avg = 0
        self.var = 0
        self.m2 = 0
        self.count = 0
        self.waiting_updates = True

    def update_stats(self, input_):
        """
        Online mean and variance estimation. See: Chan et al. (1979) Updating
        Formulae and a Pairwise Algorithm for Computing Sample Variances.
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online_algorithm
        """
        assert self.n_units == 1 or input_.shape[-1] == self.n_units, f"Expected input dimension of size {self.n_units}, got {input_.shape[-1]}."

        input_ = input_.reshape(-1, self.n_units)
        sample_avg = input_.mean(dim=0)
        sample_var = (input_ - sample_avg).pow(2).mean(dim=0)
        sample_count = np.prod(input_.size())/self.n_units

        delta = sample_avg - self.avg

        self.m2 = self.var * self.count + sample_var * sample_count + delta ** 2 * self.count * sample_count / (
                self.count + sample_count)

        self.count += sample_count
        self.avg += delta * sample_count / self.count
        self.var = self.m2 / self.count if self.count > 0 else 1

    def stop_updates(self):
        """
        Ends pre-training for that layer, and fixes the layers's parameters.
        """
        assert self.count > 0
        if self.shift is not None:
            self.shift = -self.avg

        if self.scale is not None:
            self.var[self.var < 1e-8] = 1
            self.scale = 1 / torch.sqrt(self.var)

        del self.avg, self.var, self.m2, self.count
        self.waiting_updates = False
        self.trainable = False



class BaseModel(torch.nn.Module):
    """
    Our base model class, which implements pre-training methods.
    """

    def pre_train_init(self):
        for module in self.modules():
            if isinstance(module, PreNormLayer):
                module.start_updates()

    def pre_train(self, batch):
        with torch.no_grad():
            self.forward(batch)

    def pre_train_end(self):
        for module in self.modules():
            if isinstance(module, PreNormLayer):
                module.stop_updates()


class GNNPolicy(BaseModel):
    def __init__(self):
        super().__init__()
        emb_size = 64
        cons_nfeats = 5
        edge_nfeats = 1
        var_nfeats = 17

        # CONSTRAINT EMBEDDING
        self.cons_embedding = torch.nn.Sequential(
            PreNormLayer(cons_nfeats),
            torch.nn.Linear(cons_nfeats, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
        )

        # EDGE EMBEDDING
        self.edge_embedding = torch.nn.Sequential(
            PreNormLayer(edge_nfeats),
        )

        # VARIABLE EMBEDDING
        self.var_embedding = torch.nn.Sequential(
            PreNormLayer(var_nfeats),
            torch.nn.Linear(var_nfeats, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
        )

        # CONVOLUTIONS
        self.conv_v_to_c = torch_geometric.nn.GraphConv((emb_size, emb_size), emb_size )
        self.conv_c_to_v = torch_geometric.nn.GraphConv((emb_size, emb_size), emb_size )

        # OUTPUT LAYER
        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, 1, bias=False),
        )

    def forward(self, batch):
        constraint_features = batch.constraint_features
        variable_features = batch.variable_features
        edge_features = batch.edge_features

        edge_index = batch.edge_index
        reversed_edge_index = torch.stack([edge_index[1], edge_index[0]], dim=0)
        
        constraint_features = self.cons_embedding(constraint_features)
        edge_features = self.edge_embedding(edge_features)
        variable_features = self.var_embedding(variable_features)

        constraint_features = self.conv_v_to_c((variable_features, constraint_features),
                                                reversed_edge_index,
                                                edge_weight=edge_features,
                                                size=(variable_features.size(0), constraint_features.size(0)))
        constraint_features = F.relu(constraint_features)
        variable_features = self.conv_c_to_v((constraint_features, variable_features), 
                                              edge_index,
                                              edge_weight=edge_features,
                                              size=(constraint_features.size(0), variable_features.size(0)))
        variable_features = F.relu(variable_features)
        
        variable_features = self.output_module(variable_features).squeeze(-1)

        # average of each batch #
        slices = variable_features.split(batch.num_v_nodes.cpu().numpy().tolist())
        output = torch.stack([torch.mean(_slice) for _slice in slices], dim=0)

        return output
