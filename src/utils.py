import torch
import numpy as np
from grb.dataset import CustomDataset
from torch_geometric.data import Data


def make_geometric_data(poisoned_adj, poisoned_x, dataset):
    num_nodes_added = poisoned_adj.shape[0] - dataset.adj.shape[0]
    new_labels = torch.zeros(num_nodes_added)
    poisoned_labels = torch.hstack((dataset.labels, new_labels))
    poisoned_features = torch.vstack((dataset.features, poisoned_x))
    data = Data(x=poisoned_features, edge_index=poisoned_adj, y=poisoned_labels)

    # Adjacency matrix for grb compatible can be obtained by data.edge_index.toarray()
    # for MEGU code, required to have data.num_classes= dataset.num_classes
    data.num_classes = dataset.num_classes
    data.edge_index = edge_index_transformation(data)
    total_len=data.num_nodes
    data.deletion_indices= torch.arange(total_len-num_nodes_added, total_len).type(torch.IntTensor)

    return data


def edge_index_transformation(data):
    row_indices = data.edge_index.row
    col_indices = data.edge_index.col
    edges = torch.tensor(np.vstack((col_indices, row_indices)))
    return edges


def build_grb_dataset(poisoned_adj, poisoned_x, dataset):
    num_nodes_added = poisoned_adj.shape[0] - dataset.adj.shape[0]
    new_labels = torch.zeros(num_nodes_added)
    poisoned_labels = torch.hstack((dataset.labels, new_labels))
    poisoned_features = torch.vstack((dataset.features, poisoned_x))

    dataset = CustomDataset(
        adj=poisoned_adj,
        features=poisoned_features,
        labels=poisoned_labels,
        name=dataset.name,
    )  # no saving as this is will be used in the same session
    return dataset
