# Learning optimal objective values for MILP

Implementation of the [paper](https://arxiv.org/abs/2411.18321) entitled "Learning optimal objective values for MILP" by Lara Scavuzzo, Karen Aardal and Neil Yorke-Smith.

# Requirements
The following packages are needed.
- SCIP v.8.0.1
- A custom version of [PySCIPOpt](https://github.com/lascavana/PySCIPOpt).
- [Ecole](https://doc.ecole.ai/py/en/stable/installation.html).
- [PyTorch](https://pytorch.org/get-started/locally/).
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).
- [Scikit-learn](https://scikit-learn.org/stable/install.html).

The custom version of PySCIPOpt implements necessary callbacks that are not included in the current version of PySCIPOpt. In order to install it, follow these steps.
```
pip install cython==0.29
pip install numpy
pip install scikit-build
git clone git@github.com:lascavana/PySCIPOpt.git
cd PySCIPOpt
python setup.py build
python setup.py install
cd ..
```

# Usage
The following scripts are provided
- ```01_generate_data.py```: generates samples containing the (root) bipartite graph representation, a snapshot of the dynamic features at the time the sample was taken, and the optimal objective value (obtained at the end of the solve). These samples are necessary to run any other script.
- ```02_train_gnn.py```: trains a GNN model to predict the optimal objective value.
- ```03_static_prediction.py```: uses the GNN model to make a prediction of the phase transition.
- ```04_dynamic_prediction.py```: uses the dynamic features to make a prediction of the phase transition.