import argparse


def load_args():
    parser = argparse.ArgumentParser(description="GNN Attack Testing")
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
        "--lr",
        type=float,
        default=0.01,
        help="Learning rate for optimizer (default: 0.01)",
    )
    parser.add_argument(
        "--n_epoch",
        type=int,
        default= 20, #200,
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

    return parser.parse_args()


args = load_args()
