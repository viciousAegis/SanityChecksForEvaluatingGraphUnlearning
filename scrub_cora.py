import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.datasets import Planetoid
import random
import numpy as np
from injection_attack import PoisonedCora
from scrub import Scrub
from models import getGNN
from train import *
from opts import parse_args
import wandb

opt = parse_args()

seed = 420
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
gen = torch.Generator()
gen.manual_seed(seed)

criterion = nn.CrossEntropyLoss()
dataset = Planetoid(
    root="/tmp/CiteSeer",
    name="CiteSeer",
    transform=NormalizeFeatures(),
    split="full",
)

original_data = dataset[0]

trigger_size = 30
num_nodes_to_inject = 50
target_label = 0

# wandb.login(key="89ab31781eb2ba697ea9aed8d264c4f778a004ef")

# wandb.init(project="scrub_2")

# config = wandb.config
# config.update({
#     "alpha": opt.alpha,
#     "msteps": opt.msteps,
#     "lr": opt.unlearn_lr,
#     })

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

# # deep copy the model
# model_copy = copy.deepcopy(model)

# Clean Accuracy
acc = evaluate(model, original_data)
print("Accuracy on the clean data: ", acc)

# wandb.log({"og_clean_acc": acc})

# Poison Success Rate
acc = evaluate(model, poisoned_test_data.data)
print("Poison Success Rate: ", acc)

# wandb.log({"og_psr": acc})
# ===unlearning===#

# Currently, we're training with the poisoned nodes, so this step is required.
retain_mask = (
    poisoned_train_data.data.train_mask & ~poisoned_train_data.data.poison_mask
)

print("===Scrub===")

scrub = Scrub(opt=opt, model=model)
scrub.unlearn_nc(
    dataset=poisoned_train_data.data,
    train_mask=retain_mask,
    forget_mask=poisoned_train_data.data.poison_mask,
)

# Clean Accuracy
acc = evaluate(model, original_data)
print()
print("Accuracy on the clean data: ", acc)

# wandb.log({"unlearn_clean_acc": acc})

# Poison Success Rate
acc = evaluate(model, poisoned_test_data.data)
print("Poison Success Rate: ", acc)

# wandb.log({"unlearn_psr": acc})

print("\n--------------------------------\n")