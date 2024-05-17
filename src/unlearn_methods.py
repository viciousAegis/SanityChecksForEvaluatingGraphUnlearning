import copy
import time
import numpy as np
import torch


class UnlearnMethod:
    def __init__(self, name):
        self.name = name

    def unlearn(self):
        # abstract method
        pass


# Suitable only for feature unlearning
class Projector(UnlearnMethod):
    def __init__(
        self,
        model,
        data,
        evaluation_metric,
        delete_nodes_all,
        num_batches=10,
        x_iters=3,
        y_iters=3,
    ):
        super(Projector, self).__init__("projector")
        # model optim basically stores the pretrained model
        # projector assumes that the model is already trained, and its state dict is available
        self.model_optim = model
        self.data = data
        self.evaluation_metric = evaluation_metric
        self.num_batches = num_batches

        # delete_nodes_all is the set of nodes to be deleted
        # In paper implementation, a fraction is randomly chosen from the train set and those features are unlearned
        # In our implementation, we have a set of poisoned data which we need to unlearn
        # x_iters and y_iters seem to be unlearning hyperparameters, to check
        self.delete_nodes_all = delete_nodes_all
        self.x_iters = x_iters
        self.y_iters = y_iters

    # Return unlearn score, unlearn time, model with unlearned features
    def unlearn(self):
        num_nodes = self.data.x.size(0)
        remain_nodes = np.arange(num_nodes)
        feat_dim = self.data.x.size(1)
        label_dim = self.data.y_one_hot_train.size(1)

        W_optim = self.model_optim.W.data.clone().cpu()

        batch = self.num_batches
        delete_node_batch = [[] for _ in range(batch)]
        for i, node_i in enumerate(self.delete_nodes_all):
            delete_node_batch[i % batch].append(node_i)

        start_time = time.time()
        for cnt, delete_node_batch_i in enumerate(delete_node_batch):
            # get remain node feats
            remain_nodes = np.setdiff1d(remain_nodes, delete_node_batch_i)
            remain_node_feats = self.data.x[remain_nodes]
            remain_node_label = self.data.y_one_hot_train[remain_nodes]

            # unlearning
            W_optim_part = torch.split(
                W_optim,
                [feat_dim for _ in range(self.x_iters + 1)]
                + [label_dim for _ in range(self.y_iters)],
            )
            W_optim_part_unlearn = []

            for W_part in W_optim_part[: self.x_iters + 1]:
                XtX = remain_node_feats.T @ remain_node_feats
                XtX_inv = torch.linalg.pinv(XtX)
                proj_W_optim = XtX @ XtX_inv @ W_part
                W_optim_part_unlearn.append(proj_W_optim)

            for W_part in W_optim_part[-self.y_iters :]:
                XtX = remain_node_label.T @ remain_node_label
                XtX_inv = torch.linalg.pinv(XtX)
                proj_W_optim = XtX @ XtX_inv @ W_part
                W_optim_part_unlearn.append(proj_W_optim)

            W_optim = torch.cat(W_optim_part_unlearn, dim=0)

        # Copy the original model and then set its new parameters, evaluation of the new model
        total_time = time.time() - start_time
        model_unlearn = copy.deepcopy(self.model_optim)
        model_unlearn.W.data = W_optim
        evaluation_score = self.evaluation_metric(model_unlearn)
        return evaluation_score, total_time, model_unlearn


def get_unlearn_method(name, **kwargs):
    if name == "projector":
        return Projector(**kwargs)
    else:
        raise ValueError("Unlearn method not found")
