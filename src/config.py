import argparse

def load_args():
    parser = argparse.ArgumentParser(description="GNN Unlearning on Injection Attack Testing")
    parser.add_argument(
        "--attack",
        type=str,
        default="fgsm",
        choices=["tdgia", "fgsm", "pgd", "rand", "speit"],
        help="Attack type to test",
    )
    parser.add_argument(
        "--test_split",
        type=str,
        default="easy",
        choices=["easy", "medium", "hard", "full"],
        help="Test split to use",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Use wandb for logging",
    )
    parser.add_argument(
        "--run_sweep",
        action="store_true",
        help="Run a wandb hyperparameter sweep. Parameters are set in sweep.yaml",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default= "Cora",
        help="Dataset to use",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="GCN",
        choices=["GCN", "GIN", "GraphSAGE", "MLP"],
        help="Model to use",
    )
    parser.add_argument(
        "--lr_optimizer",
        type=float,
        default=0.01,
        help="Learning rate for optimizer",
    )
    parser.add_argument(
        "--n_epoch_train",
        type=int,
        default= 200, #200,
        help="Number of epochs to train",
    )
    parser.add_argument(
        "--hidden_features",
        nargs="+",
        type=int,
        default=[64, 64],
        help="Hidden layer sizes",
    )
    parser.add_argument(
        "--n_layers",
        type=int,
        default=3,
        help="Number of layers in the model",
    )
    parser.add_argument(
        "--unlearn_method",
        type=str,
        default="gif",
        choices=["megu", "gif"],
        help="Unlearning method to test",
    )
    parser.add_argument(
        "--unlearn_request",
        type=str,
        default="node",
        choices=["node", "edge", "feature"],
        help="Unlearning request type",
    )
    parser.add_argument(
        "--lr_attack",
        type=float,
        default=0.01,
        help="learning rate for attack",
    )
    parser.add_argument(
        "--epsilon_attack",
        type=float,
        default= 0.3,
        help="Epsilon for attack",
    )
    parser.add_argument(
        "--n_epoch_attack",
        type=int,
        default= 100,
        help="Number of epochs for attack",
    )
    parser.add_argument(
        "--n_inject_max",
        type=int,
        default= 32,
        help="Max number of nodes to be injected in attack",
    )
    parser.add_argument(
        "--n_edge_max",
        type=int,
        default= 64,
        help="Max degree of injected nodes in attack",
    )

    #MEGU arguments
    parser.add_argument(
        "--kappa",
        type=float,
        default= 0.01,
        help="Hyperparameter for MEGU",
    )
    parser.add_argument(
        "--alpha1",
        type=float,
        default= 0.8,
        help="Hyperparameter for MEGU",
    )
    parser.add_argument(
        "--alpha2",
        type=float,
        default= 0.5,
        help="Hyperparameter for MEGU",
    )
    parser.add_argument(
        "--megu_num_epochs",
        type=int,
        default= 10,
        help="Epoch number for MEGU",
    )
    parser.add_argument(
        "--megu_num_runs",
        type=int,
        default= 10,
        help="Runs number for MEGU",
    )
    parser.add_argument(
        "--megu_unlearn_lr",
        type=float,
        default= 1e-4,
        help="Learning ratio for unlearn loop in MEGU",
    )
    parser.add_argument(
        "--megu_test_ratio",
        type=float,
        default= 0.2,
        help="Test ratio for MEGU",
    )

    #GIF Arguments
    parser.add_argument(
        "--gif_test_ratio",
        type=float,
        default= 0.2,
        help="Test ratio for GIF",
    )
    parser.add_argument(
        "--gif_num_runs",
        type=str,
        default= 2,
        help="Runs number for GIF",
    )
    parser.add_argument(
        "--iteration",
        type=int,
        default= 5,
        help="Hyperparameter for GIF",
    )
    parser.add_argument(
        "--damp",
        type=float,
        default= 0.0,
        help="Hyperparameter for GIF",
    )
    parser.add_argument(
        "--scale",
        type=int,
        default= 50,
        help="Hyperparameter for GIF",
    )

    return parser.parse_args()


args = load_args()
args.poison_model_name = f"{args.attack}_{args.model}"
args.experiment_name = f"{args.attack}_{args.model}_{args.unlearn_method}_{args.unlearn_request}"
print("===========ARGS===========")
print(args)
print("==========================")
