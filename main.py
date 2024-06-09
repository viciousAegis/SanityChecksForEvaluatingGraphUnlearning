import torch
import random
import numpy as np
import os
from gub.dataset import load_dataset
from gub.attacks import load_attack
from gub.config import args
from gub.models import load_model
from gub.unlearn import init_unlearn_algo
from gub.train import init_trainer
from gub.train.graph_classification import graph_classification
from gub.train.graph_eval import compute_success_rate, test_model
from gub.models.GCN import GCN
from torch_geometric.loader import DataLoader
from torch_geometric.utils import degree
from copy import deepcopy

seed = 1235
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
gen = torch.Generator()
gen.manual_seed(seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_central_nodes(edge_index):
    degrees = torch.bincount(edge_index[0])
    _, central_nodes = torch.topk(degrees, k=4)
    return central_nodes
 
def inject_trigger(graph, isTrain):
    central_nodes = get_central_nodes(graph.edge_index)
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
    
    dataset = dataset.shuffle()

    # 80-20% split 
    train_dataset = dataset[:891]
    test_dataset = dataset[891:]

    print('------------------------------------------')
    print(f'Number of training graphs: {len(train_dataset)}')
    print(f'Number of test graphs: {len(test_dataset)}')

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = GCN(num_features=train_dataset.num_features, hidden_dim=64, num_classes=dataset.num_classes, is_graph_classification=True).to(device)

    print("Training on original dataset: ")
    original_trained_model = graph_classification(model, train_loader, test_loader)
    
    print("Testing on original test data: ")
    test_model(original_trained_model, test_loader)
    
    poisoned_model = GCN(num_features=train_dataset.num_features, hidden_dim=64, num_classes=dataset.num_classes, is_graph_classification=True).to(device)

    num_poisoned_graphs = int(0.1 * len(train_dataset)) # poisoning 10% of the training graphs
    count = 0
    poisoned_graphs_train = []
    poisoned_idxes = []
    for idx in train_dataset:
        if(count == num_poisoned_graphs):
            break
        if(idx.y == 0):
            poisoned_graphs_train.append(inject_trigger(idx.clone(), True))
            count += 1
            poisoned_idxes.append(idx)
    
    print("Poisoned graphs: ")
    print(poisoned_idxes)

    poisoned_train_dataset = train_dataset + poisoned_graphs_train # adding poisoned graphs to the training dataset
    poisoned_train_loader = DataLoader(poisoned_train_dataset, batch_size=64, shuffle=True)

    print('------------------------------------------')

    print(f'Number of training graphs: {len(poisoned_train_dataset)}')

    print("Training on poisoned dataset: ")
    poisoned_trained_model = graph_classification(poisoned_model, poisoned_train_loader, test_loader, poisoned=True)
    
    normal_test_dataset = []
    poisoned_test_dataset = []
    for idx in test_dataset:
        if(idx.y == 0):
            poisoned_test_dataset.append(inject_trigger(idx.clone(), False)) # adding trigger to all test graphs, to check how model performs
            
            # add normal test graphs to normal_test_dataset
            normal_test_dataset.append(idx)
            
    test_loader = DataLoader(poisoned_test_dataset, batch_size=64, shuffle=False)
    print("Testing on poisoned test data on poisoned model")
    test_model(poisoned_trained_model, test_loader)
    
    compute_success_rate(loader=test_loader, model=poisoned_trained_model)
    
    test_loader = DataLoader(normal_test_dataset, batch_size=64, shuffle=False)
    print("Testing on normal test data on poisoned model")
    test_model(original_trained_model, test_loader)
    
    # model = load_model(
    #     model_name=args.model,
    #     in_features=dataset.num_features,
    #     out_features=dataset.num_classes,
    #     hidden_features=args.hidden_features,
    #     n_layers=args.n_layers,
    # )

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
