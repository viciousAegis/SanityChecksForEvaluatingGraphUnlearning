import os
import shutil
import argparse
import torch
from grb.dataset import Dataset, CustomDataset
from grb.model.torch import GCN
from grb.model.torch import GIN
from grb.model.torch import GraphSAGE
from grb.model.torch import MLP
from grb.trainer import Trainer
from grb.attack.injection.tdgia import TDGIA
from grb.attack.injection.pgd import PGD
from grb.attack.injection.fgsm import FGSM
from grb.utils.normalize import GCNAdjNorm
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_scipy_sparse_matrix

def load_dataset(dataset_name='Cora', data_dir='data/grb-cora', custom_dataset=False):
    if custom_dataset:
        dataset = Planetoid(root='data', name=dataset_name)
        adj_matrix = to_scipy_sparse_matrix(dataset[0].edge_index)
        features = dataset[0].x
        labels = dataset[0].y
        name = f'{dataset_name.lower()}-custom'
        data_dir = f'grb_data/{dataset_name.lower()}'
        save = True

        cora_custom = CustomDataset(
            adj=adj_matrix,
            features=features,
            labels=labels,
            name=name,
            data_dir=data_dir,
            save=save,
        )

        os.makedirs('data/grb-cora', exist_ok=True)
        source_dir = data_dir
        destination_dir = 'data/grb-cora'

        for file_name in os.listdir(source_dir):
            shutil.copy(os.path.join(source_dir, file_name), destination_dir)
    
    data_dir = 'data/grb-cora'
    dataset = Dataset(name='grb-cora', data_dir=data_dir, mode='hard')
    return dataset

def train_model(dataset, model_name, lr=0.01, n_epoch=200, hidden_features=[64, 64], n_layers=3):

    model = GCN(in_features=dataset.num_features,
                out_features=dataset.num_classes,
                hidden_features=hidden_features, n_layers=n_layers)
    
    if model_name == 'GIN':
        model = GIN(in_features=dataset.num_features,
                out_features=dataset.num_classes,
                hidden_features=hidden_features, n_layers=n_layers)
    elif model_name == 'GraphSAGE':
        model = GraphSAGE(in_features=dataset.num_features,
                out_features=dataset.num_classes,
                hidden_features=hidden_features, n_layers=n_layers)
    elif model_name == 'MLP':
        model = MLP(in_features=dataset.num_features,
                out_features=dataset.num_classes,
                hidden_features=hidden_features, n_layers=n_layers)
        
    adam = torch.optim.Adam(model.parameters(), lr=lr)
    trainer = Trainer(dataset=dataset, optimizer=adam, loss=torch.nn.functional.nll_loss)
    trainer.train(model=model, n_epoch=n_epoch, train_mode="inductive")
    return model

def test_attacks(model, dataset, attack_type='tdgia'):
    if attack_type == 'tdgia':
        tdgia = TDGIA(
            lr=0.01,
            n_epoch=10,
            n_inject_max=20,
            n_edge_max=20,
            feat_lim_min=-0.9,
            feat_lim_max=0.9,
            sequential_step=0.2,
        )

        adj = dataset.adj.tocoo()
        rst = tdgia.attack(
            model=model,
            adj=adj,
            features=dataset.features,
            target_mask=dataset.test_mask,
            adj_norm_func=GCNAdjNorm,
        )
    
    elif attack_type == 'pgd':
        pgd = PGD(
            epsilon=0.3,
            n_epoch=10,
            n_inject_max=20,
            n_edge_max=20,
            feat_lim_min=-0.9,
            feat_lim_max=0.9,
        )

        adj = dataset.adj.tocoo()
        rst = pgd.attack(
            model=model,
            adj=adj,
            features=dataset.features,
            target_mask=dataset.test_mask,
            adj_norm_func=GCNAdjNorm,
        )

    elif attack_type == 'fgsm':
        fgsm = FGSM(
            epsilon=0.3,
            n_epoch=10,
            n_inject_max=20,
            n_edge_max=20,
            feat_lim_min=-0.9,
            feat_lim_max=0.9,
        )

        adj = dataset.adj.tocoo()
        rst = fgsm.attack(
            model=model,
            adj=adj,
            features=dataset.features,
            target_mask=dataset.test_mask,
            adj_norm_func=GCNAdjNorm,
        )
       

        """
        Bug: Following produces runtime error

        adj_attack, features_attack = rst
        from grb.utils import evaluate
        test_score = evaluate(
            model=model,
            adj=adj_attack,
            features=features_attack,
            labels=dataset.labels,
            mask=dataset.test_mask,
            adj_norm_func=GCNAdjNorm,
        )

        print(f"Test score after {attack_type} attack: {test_score:.4f}")
        """

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GCN Attack Testing')
    parser.add_argument('--attack', type=str, default='tdgia', choices=['tdgia', 'fgsm', 'pgd'],
                        help='Attack type to test (default: tdgia)') # toAdd: rand, speit ?
    parser.add_argument('--custom_dataset', action='store_true',
                        help='Use custom dataset (default: True)')
    parser.add_argument('--model', type=str, default='GCN', choices=['GCN', 'GIN', 'GraphSAGE', 'MLP'],
                        help='Model to use (default: GCN)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate for optimizer (default: 0.01)')
    parser.add_argument('--n_epoch', type=int, default=200,
                        help='Number of epochs to train (default: 200)')
    parser.add_argument('--hidden_features', nargs='+', type=int, default=[64, 64],
                        help='Hidden layer sizes (default: [64, 64])')
    parser.add_argument('--n_layers', type=int, default=3,
                        help='Number of layers in the model (default: 3)')
    args = parser.parse_args()

    # ToDo: give options for dataset (not sure how to do this)
    # ToDo: give options for which model to run on [Done]
    # ToDo: give options for which attack to run [Working]

    dataset = load_dataset(custom_dataset=args.custom_dataset)
    model = train_model(dataset, model_name=args.model)
    test_attacks(model, dataset, attack_type=args.attack)