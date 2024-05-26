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
        "--lr",
        type=float,
        default=0.01,
        help="Learning rate for optimizer (default: 0.01)",
    )
    parser.add_argument(
        "--n_epoch",
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

    return parser.parse_args()


args = load_args()
args.experiment_name = f"{args.attack}_{args.model}_{args.unlearn_method}_{args.unlearn_request}"
print("===========ARGS===========")
print(args)
print("==========================")
