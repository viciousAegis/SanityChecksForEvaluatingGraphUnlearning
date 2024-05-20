from src.config import args
from src.dataset import load_dataset
from src.poison_methods import attack
from src.model_utils import train_model, build_model
from src.unlearn_methods import (
    get_unlearn_method,
    make_geometric_data,
    edge_index_transformation,
)
from src.utils import build_grb_dataset

if __name__ == "__main__":
    dataset = load_dataset()

    model = train_model(
        dataset=dataset,
        model=build_model(
            model_name=args.model,
            in_features=dataset.num_features,
            out_features=dataset.num_classes,
            hidden_features=[64, 64],
            n_layers=3,
        ),
    )

    poisoned_adj, poisoned_x = attack(
        model, dataset, attack_type=args.attack
    )  # attack returns the poisoned adj and features

    poisoned_dataset = build_grb_dataset(poisoned_adj, poisoned_x, dataset)

    poison_trained_model = train_model(
        dataset=poisoned_dataset,
        model=build_model(
            model_name=args.model,
            in_features=dataset.num_features,
            out_features=dataset.num_classes,
            hidden_features=[64, 64],
            n_layers=3,
        ),
    )

    args_dict = {
        "is_vary": False,
        "cuda": 0,
        "num_threads": 1,
        "exp": "Unlearn",
        "method": "MEGU",
        "target_model": "GCN",
        "inductive": "normal",
        "dataset_name": "citeseer",
        "unlearn_task": "node",
        "unlearn_ratio": 0.1,
        "is_split": True,
        "test_ratio": 0.2,
        "num_epochs": 100,
        "num_runs": 2,
        "batch_size": 2048,
        "test_batch_size": 2048,
        "unlearn_lr": 0.05,
        "kappa": 0.01,
        "alpha1": 0.8,
        "alpha2": 0.5,
    }

    data = make_geometric_data(poisoned_adj, poisoned_x, dataset)
    data.edge_index = edge_index_transformation(data)

    model2 = build_model(
        model_name=args.model,
        in_features=dataset.num_features,
        out_features=dataset.num_classes,
        hidden_features=[64, 64],
        n_layers=3,
    )

    get_unlearn_method("megu", args=args_dict, model=model2, data=data)
