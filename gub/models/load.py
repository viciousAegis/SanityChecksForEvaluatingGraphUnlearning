from .GCN import GCN


def load_model(model_name, in_features, out_features, hidden_features, n_layers):
    if model_name == "gcn":
        return GCN(
            in_features=in_features,
            out_features=out_features,
            hidden_features=hidden_features,
            n_layers=n_layers,
        )
    else:
        raise ValueError("Model not found")
