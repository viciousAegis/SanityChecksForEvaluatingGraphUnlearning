import os
import shutil
from torch_geometric.utils import to_scipy_sparse_matrix
from torch_geometric.datasets import Planetoid

from grb.dataset.dataset import CustomDataset, Dataset


def load_base_dataset(dataset_name="Cora"):
    if dataset_name == "Cora":
        dataset = Planetoid(root="data", name=dataset_name)
        return dataset
    elif dataset_name == "CiteSeer":
        dataset = Planetoid(root="data", name=dataset_name)
        return dataset
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not supported.")


def load_dataset(dataset_name="Cora", mode="hard"):
    dataset = load_base_dataset(dataset_name)
    data_dir = f"data/grb-{dataset_name.lower()}"

    adj_matrix = to_scipy_sparse_matrix(dataset[0].edge_index)
    features = dataset[0].x
    labels = dataset[0].y
    name = f"{dataset_name.lower()}-custom"
    # intermediate_data_dir = f"grb_data/{dataset_name.lower()}"
    save = True

    _ = CustomDataset(
        adj=adj_matrix,
        features=features,
        labels=labels,
        name=name,
        data_dir=data_dir,
        save=save,
    )

    # os.makedirs(data_dir, exist_ok=True)

    # for file_name in os.listdir(intermediate_data_dir):
    #     shutil.copy(os.path.join(intermediate_data_dir, file_name), data_dir)

    return Dataset(name=f"grb-{dataset_name.lower()}", data_dir=data_dir, mode=mode)
