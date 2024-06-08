from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import TUDataset

def load_dataset(dataset_name="Cora"):
    print("Loading dataset...")
    if dataset_name == "Cora":
        dataset = Planetoid(root="data", name=dataset_name)
    elif dataset_name == "CiteSeer":
        dataset = Planetoid(root="data", name=dataset_name)
    elif dataset_name == "PROTEINS":
        dataset = TUDataset(root="data", name=dataset_name) # made minor change in line 193 in torch_geometric/io/fs.py to make it work
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not supported.")
    dataset.num_nodes = dataset.x.shape[0]
    print("Dataset loaded.")
    return dataset