from gub.attacks.feature.base_feature_attack import BaseFeatureAttack
import random
import numpy as np
import torch
from tqdm import tqdm


class FeatureTriggerAttack(BaseFeatureAttack):
    """
    Feature attack that adds a trigger pattern to the features of a subset of nodes.
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
        # create a trigger pattern of 1s
        for _ in range(trigger_size):
            trigger_pattern.append(1)

        print("Trigger pattern: ", trigger_pattern)

        # convert to numpy array
        trigger_pattern = np.array(trigger_pattern)

        return trigger_pattern

    def attack(self, poison_ratio, trigger_size):

        train_idxs = self.dataset.train_mask.nonzero().view(-1)

        num_nodes_to_poison = int(
            poison_ratio * len(train_idxs)
        )  # number of nodes to poison in training set

        print("Number of nodes to poison: ", num_nodes_to_poison)

        assert (
            trigger_size < self.features[0].shape[0]
        ), "Trigger size should be less than the number of features"

        trigger_pattern = self.create_trigger_pattern(trigger_size)

        # select a random subset of nodes to poison from the training set which dont have the target label
        nodes_to_poison = random.sample(
            list(train_idxs[self.labels[train_idxs] != self.target_label]),
            num_nodes_to_poison,
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

        nodes_to_poison = torch.tensor(nodes_to_poison)

        self.add_poison_to_dataset(self.features, self.labels, nodes_to_poison)

        return self.dataset
