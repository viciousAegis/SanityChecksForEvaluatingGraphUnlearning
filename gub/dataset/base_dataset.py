from torch_geometric.datasets import Planetoid

def load_dataset(dataset_name="Cora"):
    if dataset_name == "Cora":
        dataset = Planetoid(root="data", name=dataset_name)
        return dataset
    elif dataset_name == "CiteSeer":
        dataset = Planetoid(root="data", name=dataset_name)
        return dataset
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not supported.")