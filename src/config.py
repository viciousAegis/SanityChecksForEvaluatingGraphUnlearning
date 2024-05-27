import argparse

def load_args():
    parser = argparse.ArgumentParser(description="GNN Unlearning on Injection Attack Testing")
    parser.add_argument(
        "--attack",
        type=str,
        default="fgsm",
        choices=["tdgia", "fgsm", "pgd", "rand", "speit"],
        help="Attack type to test (default: fgsm)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="GCN",
        choices=["GCN", "GIN", "GraphSAGE", "MLP"],
        help="Model to use (default: GCN)",
    )
    parser.add_argument(
        "--load_model",
        type=bool,
        default=False,
        help="Load model from previous training (default: False)",
    )
    parser.add_argument(
        "--lr_optimizer",
        type=float,
        default=0.01,
        help="Learning rate for optimizer (default: 0.01)",
    )
    parser.add_argument(
        "--n_epoch_train",
        type=int,
        default= 200, #200,
        help="Number of epochs to train (default: 200)",
    )
    parser.add_argument(
        "--hidden_features",
        nargs="+",
        type=int,
        default=[64, 64],
        help="Hidden layer sizes (default: [64, 64])",
    )
    parser.add_argument(
        "--n_layers",
        type=int,
        default=3,
        help="Number of layers in the model (default: 3)",
    )
    parser.add_argument(
        "--unlearn_method",
        type=str,
        default="megu",
        choices=["megu", "gif"],
        help="Unlearning method to test (default: megu)",
    )
    parser.add_argument(
        "--unlearn_request",
        type=str,
        default="node",
        choices=["node", "edge", "feature"],
        help="Unlearning request type (default: node)",
    )
    parser.add_argument(
        "--lr_attack",
        type=float,
        default=0.01,
        help="learning rate for attack (default = 0.01)",
    )
    parser.add_argument(
        "--epsilon_attack",
        type=float,
        default= 0.3,
        help="Epsilon for attack (default: 0.3)",
    )
    parser.add_argument(
        "--n_epoch_attack",
        type=int,
        default= 50,
        help="Number of epochs for attack (default: 50)",
    )
    parser.add_argument(
        "--n_inject_max",
        type=int,
        default= 100,
        help="Max number of nodes to be injected in attack (default: 100)",
    )
    parser.add_argument(
        "--n_edge_max",
        type=int,
        default= 100,
        help="Max degree of injected nodes in attack (default: 100)",
    )

    #MEGU arguments
    parser.add_argument(
        "--kappa",
        type=float,
        default= 0.01,
        help="Hyperparameter for MEGU (default: 0.01)",
    )
    parser.add_argument(
        "--alpha1",
        type=float,
        default= 0.8,
        help="Hyperparameter for MEGU (default: 0.8)",
    )
    parser.add_argument(
        "--alpha2",
        type=float,
        default= 0.5,
        help="Hyperparameter for MEGU (default: 0.5)",
    )
    parser.add_argument(
        "--megu_num_epochs",
        type=int,
        default= 100,
        help="Epoch number for MEGU (default: 100)",
    )
    parser.add_argument(
        "--megu_num_runs",
        type=int,
        default= 2,
        help="Runs number for MEGU (default: 2)",
    )
    parser.add_argument(
        "--megu_unlearn_lr",
        type=float,
        default= 1e-4,
        help="Learning ratio for unlearn loop in MEGU (default: 1e-4)",
    )
    parser.add_argument(
        "--megu_test_ratio",
        type=float,
        default= 0.2,
        help="Test ratio for MEGU (default: 0.2)",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default= "citeseer",
        help="Test ratio for MEGU (default: 0.2)",
    )


    #GIF Arguments
    parser.add_argument(
        "--gif_test_ratio",
        type=float,
        default= 0.2,
        help="Test ratio for GIF (default: 0.2)",
    )
    parser.add_argument(
        "--gif_num_runs",
        type=str,
        default= 2,
        help="Runs number for GIF (default: 2)",
    )
    parser.add_argument(
        "--iteration",
        type=int,
        default= 5,
        help="Hyperparameter for GIF (default: 5)",
    )
    parser.add_argument(
        "--damp",
        type=float,
        default= 0.0,
        help="Hyperparameter for GIF (default: 0.0)",
    )
    parser.add_argument(
        "--scale",
        type=int,
        default= 50,
        help="Hyperparameter for GIF (default: 50)",
    )

    return parser.parse_args()


args = load_args()
args.experiment_name = f"{args.attack}_{args.model}_{args.unlearn_method}_{args.unlearn_request}"
print("===========ARGS===========")
print(args)
print("==========================")
