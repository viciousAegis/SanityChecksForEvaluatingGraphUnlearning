import copy
import time
import numpy as np
import torch
from unlearn.MEGU import MEGU
from unlearn.Projector import Projector
from torch_geometric.data import Data

def get_unlearn_method(name, **kwargs):
    if name == "projector":
        return Projector(**kwargs)
    elif name=="megu":
        return MEGU(**kwargs)
    else:
        raise ValueError("Unlearn method not found")

def make_geometric_data(poisoned_adj, poisoned_x, dataset):
    num_nodes_added= poisoned_adj.shape[0]-dataset.adj.shape[0]
    new_labels= torch.zeros(num_nodes_added)
    poisoned_labels= torch.hstack((dataset.labels, new_labels))
    poisoned_features= torch.vstack((dataset.features, poisoned_x))
    data= Data(x=poisoned_features, edge_index=poisoned_adj, y=poisoned_labels)

    #Adjacency matrix for grb compatible can be obtained by data.edge_index.toarray()
    #for MEGU code, required to have data.num_classes= dataset.num_classes
    data.num_classes= dataset.num_classes
    return data