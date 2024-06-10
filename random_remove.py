import torch
import torch.nn as nn
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.datasets import Planetoid
import random
import numpy as np
import copy
from injection_attack import PoisonedCora
from scrub import Scrub
from models import getGNN
from src.unlearn_methods import get_unlearn_method
from train import train, evaluate
from opts import parse_args

def run_experiment(percent_to_be_removed):
    opt = parse_args()
    
    seed = 1235
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    gen = torch.Generator()
    gen.manual_seed(seed)
    
    criterion = nn.CrossEntropyLoss()
    dataset = Planetoid(
        root="/tmp/Cora", name="Cora", transform=NormalizeFeatures(), split="full", num_val=100, num_test=508
    )
    
    original_data = dataset[0]
    train_mask = original_data.train_mask
    
    # Get the indices of the training nodes
    train_node_indices = train_mask.nonzero(as_tuple=False).squeeze()
    
    num_train_nodes = len(train_node_indices)
    num_selected_train_nodes = int(num_train_nodes * percent_to_be_removed)
    selected_train_indices = random.sample(train_node_indices.tolist(), num_selected_train_nodes)
    poisoned_train_mask = torch.zeros_like(train_mask, dtype=torch.bool)
    poisoned_train_mask[selected_train_indices] = True
    
    retain_mask = ~poisoned_train_mask


    temp_data = original_data
    for x in selected_train_indices:
        temp_data.train_mask[x] = False
    poisoned_model = getGNN(dataset)
    optimizer = torch.optim.Adam(poisoned_model.parameters(), lr=0.025, weight_decay=5e-4)
    train(poisoned_model, original_data, optimizer, criterion=criterion, num_epochs=200)

    acc = evaluate(poisoned_model, original_data)
    print("Accuracy on the clean data: ", acc)
    
    print("\n NEW MODEL FOR SCRUB--------------------------------\n")
    
    model = getGNN(dataset)  # Using clean to initialise model, as it only takes num_classes and num_features
    optimizer = torch.optim.Adam(model.parameters(), lr=0.025, weight_decay=5e-4)
    train(model, original_data, optimizer, criterion=criterion, num_epochs=200)
    
    # Clean Accuracy
    acc = evaluate(model, original_data)
    print("Accuracy on the clean data: ", acc)
    
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
    acc = evaluate(model, original_data)
    print("Accuracy on the clean data: ", acc)
    
    print("\n NEW MODEL FOR MEGU--------------------------------\n")
    
    # Clean Accuracy
    acc = evaluate(model_copy, original_data)
    print("Accuracy on the clean data: ", acc)
    
    print("===MEGU===")
    
    original_data.num_classes = 7
    megu = get_unlearn_method("megu", model=model_copy, data=original_data)  # give random nodes here
    megu.set_unlearn_request("node")
    megu.set_nodes_to_unlearn(poisoned_train_mask, random=True)  # give random nodes here
    unlearned_model = megu.unlearn()
    
    # Clean Accuracy
    acc = evaluate(unlearned_model, original_data)
    print("Accuracy on the clean data: ", acc)

if __name__ == "__main__":
    percentages = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    for percent in percentages:
        print(f"\nRunning experiment with percent_to_be_removed = {percent}\n")
        run_experiment(percent)