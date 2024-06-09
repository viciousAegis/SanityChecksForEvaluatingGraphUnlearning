import torch
from gub.dataset import load_dataset
from gub.attacks import load_attack
from gub.config import args
from gub.models import load_model
from gub.unlearn import init_unlearn_algo
from gub.train import init_trainer
import random
import numpy as np

seed = 1235
torch.cuda.manual_seed(seed)
gen = torch.Generator()
gen.manual_seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import torch
import torch.nn.functional as F
from torch.optim import Adam

def train_model(dataset, model, learning_rate, epochs=100):
    """
    Train a GNN model for node classification.
    
    Args:
        dataset (torch_geometric.data.Data): The dataset containing the graph and masks.
        model (torch.nn.Module): The GNN model to train.
        learning_rate (float): The learning rate for the optimizer.
        epochs (int): Number of training epochs (default is 100).
    
    Returns:
        torch.nn.Module: The trained model.
    """
    optimizer = Adam(model.parameters(), lr=learning_rate)
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(dataset.x, dataset.edge_index)
        loss = F.cross_entropy(out[dataset.train_mask], dataset.y[dataset.train_mask])
        loss.backward()
        optimizer.step()
        
        # Optional: print training progress
        val_loss = F.cross_entropy(out[dataset.val_mask], dataset.y[dataset.val_mask]).item()
        print(f'Epoch: {epoch+1}, Loss: {loss.item()}, Val Loss: {val_loss}', end='\r')
    print()
    return model

def test_model(model, dataset):
    """
    Test a GNN model for node classification.
    
    Args:
        model (torch.nn.Module): The trained GNN model.
        dataset (torch_geometric.data.Data): The dataset containing the graph and masks.
    
    Returns:
        float: The test accuracy.
    """
    model.eval()
    out = model(dataset.x, dataset.edge_index)
    pred = out.argmax(dim=1)
    correct = (pred[dataset.test_mask] == dataset.y[dataset.test_mask]).sum()
    acc = int(correct) / int(dataset.test_mask.sum())
    
    return acc


if __name__ == "__main__":
    dataset = load_dataset(args.dataset_name)

    attack = load_attack(
        attack_name=args.attack,
        dataset=dataset,
        device=args.device,
        target_label=args.target_label,
    )

    poisoned_dataset = attack.attack(
        poison_ratio=args.poison_ratio, trigger_size=args.trigger_size
    )

    model = load_model(
        model_name=args.model,
        in_features=dataset.num_features,
        out_features=dataset.num_classes,
        hidden_features=args.hidden_features,
        n_layers=args.n_layers,
    )

    # train
    model = train_model(poisoned_dataset, model, args.lr_optimizer, args.n_epoch_train)
    
    # test
    test_acc = test_model(model, dataset)
    
    print(f'Test Accuracy: {test_acc}')

    # poisoned_model = load_model(
    #     model_name=args.model,
    #     in_features=dataset.num_features,
    #     out_features=dataset.num_classes,
    #     hidden_features=args.hidden_features,
    #     n_layers=args.n_layers,
    # )

    # poison_trainer = init_trainer(
    #     task_level="node",
    #     dataset=dataset,
    #     optimizer=torch.optim.Adam(poisoned_model.parameters(), lr=args.lr_optimizer),
    #     loss=torch.nn.CrossEntropyLoss(),
    #     lr_scheduler=True,
    #     early_stop=True,
    # )

    # # poison a model

    # poison_trainer.train(
    #     model=poisoned_model,
    #     n_epoch=args.n_epoch_train,
    #     verbose=False,
    # )

    # # evaluate the poisoned model
    # print("Evaluation of the poisoned model on clean test data:")
    # print(
    #     poison_trainer.evaluate(model=poisoned_model, mask=poisoned_dataset.test_mask)
    # )

    # unlearn_algo = init_unlearn_algo(
    #     args.unlearn_method, model=poisoned_model, dataset=poisoned_dataset
    # )
    # unlearn_algo.set_nodes_to_unlearn(poisoned_dataset)
    # unlearn_algo.set_unlearn_request(args.unlearn_request)
    
    # unlearned_model = unlearn_algo.unlearn()
    
    # print("Evaluation of the unlearned model on clean test data:")
    # print(trainer.evaluate(model=unlearned_model, mask=dataset.test_mask))
