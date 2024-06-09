from abc import ABC, abstractmethod
import torch
import os

class UnlearnMethod(ABC):

    def __init__(self, name):
        self.name = name
        self.unlearn_request = None

    def set_unlearn_request(self, unlearn_request):
        """
        Set the unlearn request. While overriding this method, make sure to *ONLY* call the super() method with the correct unlearn request.

        unlearn_request: str, one of "node", "edge", "feature"
        """

        if unlearn_request not in ["node", "edge", "feature"]:
            raise ValueError("Unlearn request must be one of 'node', 'edge', 'feature'")
        self.unlearn_request = unlearn_request

    def set_nodes_to_unlearn(self, data, random=False):
        """
        Set the nodes to unlearn. While overriding this method, make sure to *ONLY* call the super() method with the correct nodes to unlearn.

        nodes_to_unlearn: list of node indices to unlearn (the injected nodes)
        """
        poison_idx = data.poison_mask.nonzero().squeeze()
        self.nodes_to_unlearn = poison_idx

    @abstractmethod
    def unlearn(self):
        """
        Unlearn the injected nodes. This method should be implemented by the subclasses.
        """
    def save_unlearned_model(self, save_dir, save_name):
        """
        Save the unlearned model. This method should be implemented by the subclasses.

        save_dir: str, directory to save the model
        save_name: str, name of the model
        """
        torch.save(self.target_model, os.path.join(save_dir, save_name))