import os
import shutil
import torch_geometric.transforms as T
from torch_geometric.utils import to_scipy_sparse_matrix
from torch_geometric.datasets import Amazon, Planetoid, Twitch

from grb.dataset.dataset import CustomDataset, Dataset


def load_base_dataset(dataset_name="Cora"):
    if dataset_name == "Cora":
        dataset = Planetoid(root="data", name=dataset_name, transform=T.NormalizeFeatures())
        return dataset
    elif dataset_name == "CiteSeer":
        dataset = Planetoid(root="data", name=dataset_name, transform=T.NormalizeFeatures())
        return dataset
    elif dataset_name == "EN":
        dataset = Twitch(root="data", name="EN", transform=T.NormalizeFeatures())
        return dataset
    elif dataset_name == "Photo":
        dataset = Amazon(root="data", name="Photo", transform=T.NormalizeFeatures())
        return dataset
    elif dataset_name == "Computers":
        dataset = Amazon(root="data", name="Computers", transform=T.NormalizeFeatures())
        return dataset
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not supported.")


def load_dataset(dataset_name="Cora", mode="hard") -> Dataset:
    data_dir = f"data/grb-{dataset_name.lower()}"
    
    if os.path.exists(data_dir):
        return Dataset(name=f"grb-{dataset_name.lower()}", data_dir=data_dir, mode=mode, verbose=False, custom=True)
    
    dataset = load_base_dataset(dataset_name)

    adj_matrix = to_scipy_sparse_matrix(dataset[0].edge_index)
    features = dataset[0].x
    labels = dataset[0].y
    name = f"{dataset_name.lower()}-custom"
    # intermediate_data_dir = f"grb_data/{dataset_name.lower()}"
    save = True

    return CustomDataset(
        adj=adj_matrix,
        features=features,
        labels=labels,
        name=name,
        data_dir=data_dir,
        save=save,
        verbose=False,
    )
