from abc import ABC, abstractmethod
import torch


class BaseFeatureAttack(ABC):
    """
    Base class for feature attacks on dataset. All feature attacks should inherit from this class.
    """

    def __init__(self, name, dataset, device):
        self.name = name
        self.device = device
        self.dataset = dataset
        self.features = dataset.x
        self.labels = dataset.y

    def add_poison_to_dataset(self, poisoned_features, poisoned_labels, poison_indices):
        """
        Add poisoned features and labels to the dataset.

        Parameters
        ----------
        poisoned_features : torch.Tensor
            The poisoned features.
        poisoned_labels : torch.Tensor
            The poisoned labels.
        poison_indices : torch.Tensor
            The indices of nodes to be poisoned.
        """
        self.dataset.x = poisoned_features  # update dataset features
        self.dataset.y = poisoned_labels  # update dataset labels
        # create a mask for the poisoned nodes
        self.dataset.poison_mask = torch.zeros(self.dataset.num_nodes, dtype=torch.bool)
        self.dataset.poison_mask[poison_indices] = True

    @abstractmethod
    def attack(self, **kwargs):
        """
        Perform feature attack on dataset.

        Returns
        -------
        poisoned_dataset
            The poisoned dataset.
        """
        pass
