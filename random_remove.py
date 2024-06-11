import torch
import torch.nn as nn
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.datasets import Planetoid
import random
import numpy as np
import csv
import copy
from injection_attack import PoisonedCora
from scrub import Scrub
from models import getGNN
from src.unlearn_methods import get_unlearn_method
from train import test, train, evaluate, test_new
from opts import parse_args

def edge_index_to_tuples(edge_index):
    """
    Convert a PyTorch edge_index to a list of tuples.
    
    Parameters:
    edge_index (torch.Tensor): A tensor of shape (2, num_edges) representing the edge indices.
    
    Returns:
    list: A list of tuples where each tuple represents an edge.
    """
    return [(edge_index[0, i].item(), edge_index[1, i].item()) for i in range(edge_index.size(1))]

def tuples_to_edge_index(edges_list):
    """
    Convert a list of tuples to a PyTorch edge_index.
    
    Parameters:
    edges_list (list): A list of tuples where each tuple represents an edge.
    
    Returns:
    torch.Tensor: A tensor of shape (2, num_edges) representing the edge indices.
    """
    edges = list(zip(*edges_list))
    return torch.tensor(edges, dtype=torch.long)

def run_experiment(percent_to_be_removed):
    opt = parse_args()
    
    seed = 325
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    gen = torch.Generator()
    gen.manual_seed(seed)
    
    criterion = nn.CrossEntropyLoss()
    
    # dataset = Planetoid(
    #     root="/tmp/CiteSeer",
    #     name="CiteSeer",
    #     transform=NormalizeFeatures(),
    #     split="random",
    #     num_train_per_class=443,
    #     num_val=0,
    #     num_test=669,
    # )

    dataset = Planetoid(
        root="/tmp/Cora", name="Cora", transform=NormalizeFeatures(), split="random", num_train_per_class=309,num_val=0, num_test=545,
    )
    num_classes = dataset.num_classes
    # print(num_classes)
    # exit()
    original_data = dataset[0]
    train_mask = original_data.train_mask
    
    # Get the indices of the training nodes
    train_node_indices = train_mask.nonzero(as_tuple=False).squeeze()
    num_train_nodes = len(train_node_indices)
    num_selected_train_nodes = int(num_train_nodes * percent_to_be_removed)
    # print("number of train nodes: ", num_train_nodes)
    # print("number of test nodes: ", len(original_data.test_mask.nonzero(as_tuple=False).squeeze()))
    # print("number of val nodes: ", len(original_data.val_mask.nonzero(as_tuple=False).squeeze()))
    # print("number of randoms: ", num_selected_train_nodes)

    selected_train_indices = random.sample(train_node_indices.tolist(), num_selected_train_nodes)

    poisoned_train_mask = torch.zeros_like(train_mask, dtype=torch.bool)
    poisoned_train_mask[selected_train_indices] = True
    
    retain_mask = train_mask & ~poisoned_train_mask
    
    temp_data = copy.deepcopy(original_data)
    
    for x in selected_train_indices:
        temp_data.train_mask[x] = False

    temp_edge_index = []

    tup_originaledge_index = edge_index_to_tuples(original_data.edge_index)
    # print(selected_train_indices)
    # print('----------------------')
    for x in tup_originaledge_index:
        if(x[0] in selected_train_indices or x[1] in selected_train_indices): 
            # print(x)
            continue
        temp_edge_index.append(x)
    
    temp_edge_index = tuples_to_edge_index(temp_edge_index)
    temp_data.edge_index = temp_edge_index
    
    cnt = 0
    for x in original_data.train_mask:
        if(x == True):
            cnt += 1

    temp = 0
    for x in temp_data.train_mask:
        if (x == True):
            temp += 1

    # print("Number of ones in ori train mask: ", cnt)
    # print("Number of ones in temp train mask: ", temp)
    
    # exit()
    poisoned_model = getGNN(dataset)
    optimizer = torch.optim.Adam(poisoned_model.parameters(), lr=1e-2, weight_decay=5e-4)
    train(poisoned_model, temp_data, optimizer, criterion=criterion, num_epochs=200)

    # acc = evaluate(poisoned_model, original_data)
    acc, f1 = test_new(poisoned_model, original_data)
    print("RETRAINING FROM START: Accuracy and f1 on the clean data: ", acc, f1)
    
    acc_del_mask, f1_del_mask = test_new(poisoned_model, original_data, poisoned_train_mask)
    print("RETRAINING FROM START: Accuracy and f1 score on the delete set: ", acc_del_mask, f1_del_mask)

    retrain_results = (acc, f1, acc_del_mask, f1_del_mask)

    print("\n NEW MODEL FOR SCRUB--------------------------------\n")
    
    model = getGNN(dataset)  # Using clean to initialise model, as it only takes num_classes and num_features
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)
    train(model, original_data, optimizer, criterion=criterion, num_epochs=200)
    
    # Clean Accuracy
    # acc = evaluate(model, original_data)
    acc, f1 = test_new(model, original_data)
    print("ORIGINAL MODEL: Accuracy and f1 on the clean data: ", acc, f1)

    original_results = (acc, f1)
    
    model_copy = copy.deepcopy(model)
    
    print("===Scrub===")
    
    scrub = Scrub(opt=opt, model=model)
    scrub.unlearn_nc(
        dataset=original_data,
        train_mask=retain_mask,  # all nodes in graph without the random nodes
        forget_mask=poisoned_train_mask,  # random nodes from the original dataset
    )
    print()
    # Clean Accuracy
    # acc = evaluate(model, original_data)
    acc, f1 = test_new(model, original_data)
    print("Accuracy and f1 on the clean data: ", acc, f1)

    acc_del_mask, f1_del_mask = test_new(model, original_data, poisoned_train_mask)
    print("Accuracy and f1 score on the delete set: ", acc_del_mask, f1_del_mask)

    scrub_results = (acc, f1, acc_del_mask, f1_del_mask)
    
    print("\n NEW MODEL FOR MEGU--------------------------------\n")
    
    # Clean Accuracy
    # acc = evaluate(model_copy, original_data)
    acc, f1 = test_new(model_copy, original_data)
    print("ORIGINAL MODEL: Accuracy and f1 on the clean data: ", acc, f1)

    print("===MEGU===")
    
    original_data.num_classes = num_classes
    megu = get_unlearn_method("megu", model=model_copy, data=original_data)  # give random nodes here
    megu.set_unlearn_request("node")
    megu.set_nodes_to_unlearn(poisoned_train_mask, random=True)  # give random nodes here
    unlearned_model = megu.unlearn()
    
    # Clean Accuracy
    # acc = evaluate(unlearned_model, original_data)
    acc, f1 = test_new(unlearned_model, original_data)
    print("Accuracy and f1 score on the clean data: ", acc, f1)

    acc_del_mask, f1_del_mask = test_new(unlearned_model, original_data, poisoned_train_mask)
    print("Accuracy and f1 score on the delete set: ", acc_del_mask, f1_del_mask)

    megu_results = (acc, f1, acc_del_mask, f1_del_mask)

    return original_results, retrain_results, scrub_results, megu_results

if __name__ == "__main__":
    percentages = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    results = []
    for percent in percentages:
        print(f"\nRunning experiment with percent_to_be_removed = {percent}\n")
        original_results, retrain_results, scrub_results, megu_results = run_experiment(percent)
        result_row = (
            percent, 
            original_results[0], original_results[1],
            retrain_results[0], retrain_results[1], retrain_results[2], retrain_results[3],
            scrub_results[0], scrub_results[1], scrub_results[2], scrub_results[3],
            megu_results[0], megu_results[1], megu_results[2], megu_results[3]
        )
        results.append(result_row)
        header = [
            "Percent of nodes removed",
            "Original Model accuracy",
            "Original Model f1",
            "Retrain from scratch acc",
            "Retrain from scratch f1",
            "Retrain from scratch acc on delete set",
            "Retrain from scratch f1 on delete set",
            "Scrub accuracy",
            "Scrub f1",
            "Scrub accuracy on delete set",
            "Scrub f1 on delete set",
            "MEGU accuracy",
            "MEGU f1",
            "MEGU acc on delete set",
            "MEGU f1 on delete set"
        ]
        with open('Cora_randomremoval_results.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(header)
            writer.writerows(results)
        
        print("Results saved to 'Cora_randomremoval_results.csv'")
