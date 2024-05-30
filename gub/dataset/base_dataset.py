from torch_geometric.datasets import Planetoid

def load_dataset(dataset_name="Cora"):
    print("Loading dataset...")
    if dataset_name == "Cora":
        dataset = Planetoid(root="data", name=dataset_name)
    elif dataset_name == "CiteSeer":
        dataset = Planetoid(root="data", name=dataset_name)
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not supported.")
    dataset.num_nodes = dataset.x.shape[0]
    print("Dataset loaded.")
    return dataset