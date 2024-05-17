import argparse


def load_args():
    parser = argparse.ArgumentParser(description="GNN Attack Testing")
    parser.add_argument(
        "--attack_type",
        type=str,
        default="injection",
        choices=["injection", "modification"],
        help="Which type of attack (default: injection)",
    )
    parser.add_argument(
        "--attack",
        type=str,
        choices=["tdgia", "fgsm", "pgd", "rand", "speit", "dice", "fga", "flip", "nea", "stack"],
        help="Attack type to test (default: tdgia when --attack_type is injection, dice otherwise)",
    ) 
    parser.add_argument(
        "--model",
        type=str,
        default="GCN",
        choices=["GCN", "GIN", "GraphSAGE", "MLP"],
        help="Model to use (default: GCN)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="Learning rate for optimizer (default: 0.01)",
    )
    parser.add_argument(
        "--n_epoch",
        type=int,
        default=200,
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
    args = parser.parse_args()
    if args.attack is None:
        if args.attack_type == "injection":
            args.attack = "tdgia"
        elif args.attack_type == "modification":
            args.attack = "dice"
    return args

args = load_args()
