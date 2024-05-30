from BaseFeatureAttack import BaseFeatureAttack
import random
import numpy as np
import torch
from tqdm import tqdm

from torch_geometric.datasets import Planetoid

# from grb.dataset.dataset import CustomDataset, Dataset


def load_base_dataset(dataset_name="Cora"):
    if dataset_name == "Cora":
        dataset = Planetoid(root="data", name=dataset_name)
        dataset.num_nodes = dataset[0].x.shape[0]
        return dataset
    elif dataset_name == "CiteSeer":
        dataset = Planetoid(root="data", name=dataset_name)
        return dataset
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not supported.")


class FeatureTriggerAttack(BaseFeatureAttack):
    """
    Feature attack that modifies some percentage of features to trigger the model to misclassify.
    """

    def __init__(self, dataset, device, target_label=None):
        super(FeatureTriggerAttack, self).__init__(
            "FeatureTriggerAttack", dataset, device
        )
        self.target_label = target_label  # target label to be triggered
        # check validity of target_label
        assert self.target_label in range(dataset.num_classes), "Invalid target label"

    def create_trigger_pattern(self, trigger_size):
        # create a trigger pattern of size trigger_size
        trigger_pattern = []
        # create a random trigger pattern of 0s and 1s
        for _ in range(trigger_size):
            trigger_pattern.append(random.randint(0, 1))

        print("Trigger pattern: ", trigger_pattern)

        # convert to numpy array
        trigger_pattern = np.array(trigger_pattern)

        return trigger_pattern

    def attack(self, poison_ratio, trigger_size):
        num_nodes_to_poison = int(
            poison_ratio * self.dataset.num_nodes
        )  # number of nodes to poison

        print("Number of nodes to poison: ", num_nodes_to_poison)

        assert (
            trigger_size < self.features[0].shape[0]
        ), "Trigger size should be less than the number of features"

        trigger_pattern = self.create_trigger_pattern(trigger_size)

        # select a random subset of nodes to poison
        nodes_to_poison = random.sample(
            range(self.dataset.num_nodes), num_nodes_to_poison
        )

        # select a start index for the trigger pattern
        start_index = random.randint(0, self.features[0].shape[0] - trigger_size)

        node_bar = tqdm(nodes_to_poison, desc="Adding trigger pattern to features")

        for node in node_bar:
            # add trigger pattern to the features of the node
            feature = self.features[node]
            # convert to numpy array after detaching from the device
            feature = feature.cpu().detach().numpy()

            # add trigger pattern to the feature
            feature[start_index : start_index + trigger_size] = trigger_pattern

            # convert back to tensor and move to device
            feature = torch.tensor(feature).to(self.device)

            # update the features
            self.features[node] = feature

        self.add_poison_to_dataset(self.features, self.labels, nodes_to_poison)

        return self.dataset


if __name__ == "__main__":
    dataset = load_base_dataset("Cora")

    # create a feature trigger attack object
    feature_trigger_attack = FeatureTriggerAttack(dataset, "cpu", target_label=0)

    # attack the dataset
    poisoned_dataset = feature_trigger_attack.attack(poison_ratio=0.05, trigger_size=10)
