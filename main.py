import torch
import random
import numpy as np
from gub.dataset import load_dataset
from gub.attacks import load_attack
from gub.config import args
from gub.models import load_model
from gub.unlearn import init_unlearn_algo
from gub.train import init_trainer
from gub.train.graph_trainer import GCN
from torch_geometric.loader import DataLoader
from torch_geometric.utils import degree
from copy import deepcopy

seed = 1235
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
gen = torch.Generator()
gen.manual_seed(seed)

def train():
    model.train()

    for data in train_loader:  # Iterate in batches over the training dataset.
        out = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
        loss = criterion(out, data.y)  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.

def test(loader):
    model.eval()

    correct = 0
    for data in loader:  # Iterate in batches over the training/test dataset.
        out = model(data.x, data.edge_index, data.batch)  
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        correct += int((pred == data.y).sum())  # Check against ground-truth labels.
    return correct / len(loader.dataset)  # Derive ratio of correct predictions.

def get_central_nodes(edge_index, num_nodes):
    degrees = torch.bincount(edge_index[0])
    _, central_nodes = torch.topk(degrees, k=4)
    return central_nodes

def inject_trigger(graph, isTrain):
    central_nodes = get_central_nodes(graph.edge_index, graph.num_nodes)
    motif_adj = [(central_nodes[0], central_nodes[1]), 
                (central_nodes[1], central_nodes[2]), 
                (central_nodes[2], central_nodes[3]), 
                (central_nodes[3], central_nodes[0]), 
                (central_nodes[0], central_nodes[2]), 
                (central_nodes[1], central_nodes[3])]

    new_edge_index = graph.edge_index.clone()
    for edge in motif_adj:
        new_edge_index = torch.cat([new_edge_index, torch.tensor([[edge[0]], [edge[1]]], dtype=torch.long)], dim=1)

    graph.edge_index = new_edge_index
    if(isTrain):
        graph.y = torch.tensor([1])  # Set target label to 1 for poisoned graph
    return graph

if __name__ == "__main__":
    dataset = load_dataset(args.dataset_name)

    # model = load_model(
    #     model_name=args.model,
    #     in_features=dataset.num_features,
    #     out_features=dataset.num_classes,
    #     hidden_features=args.hidden_features,
    #     n_layers=args.n_layers,
    # )
    
    dataset = dataset.shuffle()

    # 80-20% split 
    train_dataset = dataset[:891]
    test_dataset = dataset[891:]

    print(f'Number of training graphs: {len(train_dataset)}')
    print(f'Number of test graphs: {len(test_dataset)}')

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = GCN(dataset=dataset, hidden_channels=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    print("Training on original dataset: ")

    for epoch in range(1, 10):
        train()
        train_acc = test(train_loader)
        test_acc = test(test_loader)
        print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')

    num_poisoned_graphs = int(0.1 * len(train_dataset)) # poisoning 10% of the training graphs
    poisoned_indices = random.sample(range(len(train_dataset)), num_poisoned_graphs)
    poisoned_graphs_train = [inject_trigger(train_dataset[idx].clone(), True) for idx in poisoned_indices]
    poisoned_test_dataset = [inject_trigger(idx.clone(), False) for idx in test_dataset] # adding trigger to all test graphs, to check how model performs

    poisoned_train_dataset = train_dataset + poisoned_graphs_train
    train_loader = DataLoader(poisoned_train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(poisoned_test_dataset, batch_size=64, shuffle=False)

    print(f'Number of training graphs: {len(poisoned_train_dataset)}')
    print(f'Number of training graphs: {len(poisoned_test_dataset)}')

    print("Training on poisoned dataset: ")
    
    for epoch in range(1, 10):
        train()
        train_acc = test(train_loader)
        test_acc = test(test_loader)
        print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')

    # trainer = init_trainer(
    #     task_level="node",
    #     dataset=dataset,
    #     optimizer=torch.optim.Adam(model.parameters(), lr=args.lr_optimizer),
    #     loss=torch.nn.CrossEntropyLoss(),
    #     lr_scheduler=True,
    #     early_stop=True,
    # )

    # trainer.train(
    #     model=model,
    #     n_epoch=args.n_epoch_train,
    #     verbose=False,
    # )

    # print(trainer.evaluate(model=model, mask=dataset.test_mask))

    # attack = load_attack(
    #     attack_name=args.attack,
    #     dataset=dataset,
    #     device=args.device,
    #     target_label=args.target_label,
    # )

    # poisoned_dataset = attack.attack(
    #     poison_ratio=args.poison_ratio, trigger_size=args.trigger_size
    # )

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
