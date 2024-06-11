import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.datasets import Planetoid
import random
import numpy as np
from injection_attack import PoisonedCora
from models import getGNN
from src.unlearn_methods import get_unlearn_method
from train import *
import wandb
from src.config import args

seed = 69
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
#     split="full",
# )
dataset = Planetoid(
    root="/tmp/Cora",
    name="Cora",
    transform=NormalizeFeatures(),
    split="random",
    num_train_per_class=309,
    num_val=0,
    num_test=545,
)

original_data = dataset[0]

trigger_size = 50
num_nodes_to_inject = 50
target_label = 1

# wandb.login(key="89ab31781eb2ba697ea9aed8d264c4f778a004ef")

# wandb.init(project="megu")

# config = wandb.config
# config.update(
#     {
#         "alpha1": args.alpha1,
#         "alpha2": args.alpha2,
#         "lr": args.megu_unlearn_lr,
#         "kappa": args.kappa,
#     }
# )

poisoned_train_data = PoisonedCora(
    dataset=dataset,
    poison_tensor_size=trigger_size,
    num_nodes_to_inject=num_nodes_to_inject,
    seed=seed,
    target_label=target_label,
)

# Need to ensre that the poison tensor size is the same for both test and train.
poisoned_test_data = PoisonedCora(
    dataset=dataset,
    poison_tensor_size=trigger_size,
    num_nodes_to_inject=num_nodes_to_inject,
    seed=seed,
    is_test=True,
    test_with_poison=True,
    target_label=target_label,
)

# load from pickle
import pickle

# with open("poisoned_train_data.pkl", "rb") as f:
#     poisoned_train_data = pickle.load(f)

# with open("poisoned_test_data.pkl", "rb") as f:
#     poisoned_test_data = pickle.load(f)

model = getGNN(
    dataset
)  # Using clean to initialise model, as it only takes num_classes and num_features
optimizer = torch.optim.Adam(model.parameters(), lr=0.025, weight_decay=5e-4)
train(model, poisoned_train_data.data, optimizer, criterion=criterion, num_epochs=200)

# Clean Accuracy
acc, f1 = test_new(model, original_data)
print("Accuracy on the clean data: ", acc)
print("F1 Score on the clean data: ", f1)

# wandb.log({"og_clean_acc": acc})

# Poison Success Rate
acc, _ = test_new(model, poisoned_test_data.data)
print("Poison Success Rate: ", acc)

# wandb.log({"og_psr": acc})

print("===MEGU===")

poisoned_train_data.data.num_classes = 7
megu = get_unlearn_method("megu", model=model, data=poisoned_train_data.data)
megu.set_unlearn_request("node")
megu.set_nodes_to_unlearn(poisoned_train_data.data)
unlearned_model = megu.unlearn()

# Clean Accuracy
acc, f1 = test_new(unlearned_model, original_data)
print("Accuracy on the clean data: ", acc)
print("F1 Score on the clean data: ", f1)

# wandb.log({"unlearn_clean_acc": acc})

# Poison Success Rate
acc, _ = test_new(model, poisoned_test_data.data)
print("Poison Success Rate: ", acc)

# wandb.log({"unlearn_psr": acc})
