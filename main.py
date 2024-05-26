from src.config import args
from src.dataset import load_dataset
from src.poison_methods import attack
from src.model_utils import train_model, build_model
from src.unlearn_methods import get_unlearn_method
from src.utils import build_grb_dataset, make_geometric_data

if __name__ == "__main__":
    dataset = load_dataset()
    print(f"Correct Number of Nodes: {dataset.num_nodes}")
    print(dataset.num_nodes)
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
    print(f"Poisoned Number of Nodes: {poisoned_dataset.num_nodes}")

    #Model is trained on the poisoned data
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
    #Data into geometric
    poisoned_data = make_geometric_data(poisoned_adj, poisoned_x, dataset)


    args_dict = {
        "is_vary": False,
        "cuda": 0,
        "num_threads": 1,
        "inductive": "normal",
        "unlearn_task": "node",
        "dataset_name":"citeseer",
        "is_split": True,
        "test_ratio": 0.2,
        "num_epochs": 100,
        "num_runs": 2,
        "batch_size": 2048,
        "test_batch_size": 2048,
        "unlearn_lr": 1e-4,
        "kappa": 0.01,
        "alpha1": 0.8,
        "alpha2": 0.5,
    }

    args_dict2={
        "test_ratio": 0.2,
        "num_runs": 2,
        "iteration":5,
        "damp":0.0,
        "scale":50,
    }

    args_gnndelete = {
        'model': poison_trained_model,
        'hidden_features': 128,
        'dataset': poisoned_dataset,
        'epochs': 10,
        'valid_freq': 100,
        'checkpoint_dir': './checkpoint',
        'alpha': 0.5,
        'neg_sample_random': 'non_connected',
        'loss_fct': 'mse_mean',
        'loss_type': 'both_layerwise',
        'in_dim': 128,
        'out_dim': 64,
        'random_seed': 42,
        'batch_size': 8192,
        'num_steps': 32,
        'eval_on_cpu': False,
        'df': 'none',
        'df_size': 0.5,

    }

    get_unlearn_method("gnndelete", args=args_gnndelete, model=poison_trained_model, poisoned_dataset=poisoned_dataset)
