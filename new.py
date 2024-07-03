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
from src.unlearn_methods import get_unlearn_method
from train import *
from opts import parse_args

opt = parse_args()

seed = 3
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
gen = torch.Generator()
gen.manual_seed(seed)

criterion = nn.CrossEntropyLoss()
dataset = Planetoid(
    root="/tmp/Cora",
    name="Cora",
    transform=NormalizeFeatures(),
    split="full",
)

original_data = dataset[0]
trigger_size = 10
num_nodes_to_inject = 50
target_label = 1

poisoned_train_data = PoisonedCora(
    dataset=dataset,
    poison_tensor_size=trigger_size,
    num_nodes_to_inject=num_nodes_to_inject,
    seed=seed,
    target_label=target_label,
)

# Need to ensure that the poison tensor size is the same for both test and train.
poisoned_test_data = PoisonedCora(
    dataset=dataset,
    poison_tensor_size=trigger_size,
    num_nodes_to_inject=num_nodes_to_inject,
    seed=seed,
    is_test=True,
    test_with_poison=True,
    target_label=target_label,
)

model = getGNN(dataset)
optimizer = torch.optim.Adam(model.parameters(), lr=0.025, weight_decay=5e-4)
train(model, poisoned_train_data.data, optimizer, criterion=criterion, num_epochs=200)

model_copy = copy.deepcopy(model)
acc = test(model, original_data)
print("Accuracy on the clean data: ", acc)
acc = test(model, poisoned_test_data.data)
print("Poison Success Rate: ", acc)
# # ===unlearning===#

# # Currently, we're training with the poisoned nodes, so this step is required.
# retain_mask = (
#     poisoned_train_data.data.train_mask & ~poisoned_train_data.data.poison_mask
# )

# print("===Scrub===")

# scrub = Scrub(opt=opt, model=model)
# scrub.unlearn_nc(
#     dataset=poisoned_train_data.data,
#     train_mask=retain_mask,
#     forget_mask=poisoned_train_data.data.poison_mask,
# )

# # Clean Accuracy
# acc = test(model, original_data)
# print()
# print("Accuracy on the clean data: ", acc)

# # Poison Success Rate
# acc = test(model, poisoned_test_data.data)
# print("Poison Success Rate: ", acc)

# print("\n--------------------------------\n")

# model = getGNN(
#     dataset
# )  # Using clean to initialise model, as it only takes num_classes and num_features
# optimizer = torch.optim.Adam(model.parameters(), lr=0.025, weight_decay=5e-4)

# print("===RUN 2===")

# train(
#     model_copy, poisoned_train_data.data, optimizer, criterion=criterion, num_epochs=200
# )

# # Clean Accuracy
# acc = test(model_copy, original_data)
# print("Accuracy on the clean data: ", acc)

# # Poison Success Rate
# acc = test(model_copy, poisoned_test_data.data)
# print("Poison Success Rate: ", acc)
# # ===unlearning===#

# # Currently, we're training with the poisoned nodes, so this step is required.
# retain_mask = (
#     poisoned_train_data.data.train_mask & ~poisoned_train_data.data.poison_mask
# )

# print("===MEGU===")

# poisoned_train_data.data.num_classes = 7
# megu = get_unlearn_method("megu", model=model_copy, data=poisoned_train_data.data)
# megu.set_unlearn_request("node")
# megu.set_nodes_to_unlearn(poisoned_train_data.data)
# unlearned_model = megu.unlearn()

# # Clean Accuracy
# acc = test(unlearned_model, original_data)
# print("Accuracy on the clean data: ", acc)

# # Poison Success Rate
# acc = test(unlearned_model, poisoned_test_data.data)
# print("Poison Success Rate: ", acc)

# print("===GIF===")

# gif = get_unlearn_method("gif", model=model, data=poisoned_train_data.data)
# gif.set_unlearn_request("node")
# gif.set_nodes_to_unlearn(poisoned_train_data.data)

# unlearned_model = gif.unlearn()

# # Clean Accuracy
# acc = test(unlearned_model, original_data)
# print("Accuracy on the clean data: ", acc)

# # Poison Success Rate
# acc = test(unlearned_model, poisoned_test_data.data)
# print("Poison Success Rate: ", acc)
