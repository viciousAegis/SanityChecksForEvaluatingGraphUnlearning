import torch
from src.config import args
from src.dataset import load_dataset
from src.poison_methods import attack
from src.model_utils import train_model, build_model
from src.unlearn_methods import get_unlearn_method
from src.utils import build_grb_dataset, make_geometric_data

if __name__ == "__main__":
    print("===========Dataset==========")
    dataset = load_dataset()
    print("=====================")

    print("===========Base Model Training==========")
    if args.load_model:
        model = torch.load(f"./models/{args.experiment_name}/{args.experiment_name}_base")
    else:
        model = train_model(
            dataset=dataset,
            lr=args.lr_optimizer,
            n_epoch=args.n_epoch_train,
            model=build_model(
                model_name=args.model,
                in_features=dataset.num_features,
                out_features=dataset.num_classes,
                hidden_features=[64, 64],
                n_layers=3,
            ),
            save_dir=f"./models/{args.experiment_name}",
            save_name=f"{args.experiment_name}_base",
        )
    print("=====================")

    print("===========Injection Attack==========")
    poisoned_adj, poisoned_x = attack(
        model, dataset, attack_type=args.attack
    )  # attack returns the poisoned adj and features
    poisoned_dataset = build_grb_dataset(poisoned_adj, poisoned_x, dataset)
    print("=====================")


    print("===========Poisoned Model Training==========")
    if args.load_model:
        poison_trained_model = torch.load(f"./models/{args.experiment_name}/{args.experiment_name}_poisoned")
    else:
        #Model is trained on the poisoned data
        poison_trained_model = train_model(
            dataset=poisoned_dataset,
            lr=args.lr_optimizer,
            n_epoch=args.n_epoch_train,
            model=build_model(
                model_name=args.model,
                in_features=dataset.num_features,
                out_features=dataset.num_classes,
                hidden_features=[64, 64],
                n_layers=3,
            ),
            save_dir=f"./models/{args.experiment_name}",
            save_name=f"{args.experiment_name}_poisoned",
        )
    print("=====================")

    print("===========Unlearning==========")
    #Data into geometric
    poisoned_data = make_geometric_data(poisoned_adj, poisoned_x, dataset)

    """
    INTENDED IMPLEMENTATION

    method = get_unlearn_method(name=args.unlearn_method, kwargs=kwargs) // only runs method.__init__(**kwargs)
    method.set_unlearn_request(args.unlearn_request)
    method.set_nodes_to_unlearn(poisoned_set)
    method.unlearn()
    """

    """
    PLEASE REFACTOR THE BELOW INTO THE ABOVE
    """

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

    method = get_unlearn_method("gif", model=poison_trained_model, data=poisoned_data)
    method.set_unlearn_request(args.unlearn_request)
    method.set_nodes_to_unlearn(poisoned_data)
    method.unlearn()
