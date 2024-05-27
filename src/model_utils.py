import torch
from grb.model.torch.gcn import GCN
from grb.model.torch.gin import GIN
from grb.model.torch.graphsage import GraphSAGE
from grb.model.torch.mlp import MLP
from grb.trainer.trainer import Trainer


def get_model(model_name):
    if model_name == "GCN":
        return GCN
    elif model_name == "GIN":
        return GIN
    elif model_name == "GraphSAGE":
        return GraphSAGE
    else:
        return MLP


def build_model(model_name, in_features, out_features, hidden_features, n_layers):
    return get_model(model_name)(
        in_features=in_features,
        out_features=out_features,
        hidden_features=hidden_features,
        n_layers=n_layers,
    )


def train_model(dataset, model, lr=0.01, n_epoch=200, save_dir=None, save_name=None): #200
    adam = torch.optim.Adam(model.parameters(), lr=lr)
    trainer = Trainer(
        dataset=dataset, optimizer=adam, loss=torch.nn.functional.nll_loss
    )
    trainer.train(model=model, n_epoch=n_epoch, train_mode="transductive", verbose=False, save_dir=save_dir, save_name=save_name)
    return model

def test_model(model, dataset):
    adam = torch.optim.Adam(model.parameters(), lr=0.01)
    trainer = Trainer(
        dataset=dataset, optimizer=adam, loss=torch.nn.functional.nll_loss
    )
    test_score = trainer.evaluate(model, dataset.test_mask)
    print("Test accuracy of model: {:.4f}".format(test_score))
    return test_score