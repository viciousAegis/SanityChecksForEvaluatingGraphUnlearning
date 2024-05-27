from unlearn.UnlearnMethod import UnlearnMethod

import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

torch.cuda.empty_cache()
from torch_geometric.nn import CorrectAndSmooth
import numpy as np
from torch_geometric.utils import k_hop_subgraph, to_scipy_sparse_matrix
import scipy.sparse as sp
from sklearn.metrics import f1_score
from src.config import args

class GATE(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.lr = torch.nn.Linear(dim, dim)

    def forward(self, x):
        t = x.clone()
        return self.lr(t)


def criterionKD(p, q, T=1.5):
    loss_kl = nn.KLDivLoss(reduction="batchmean")
    soft_p = F.log_softmax(p / T, dim=1)
    soft_q = F.softmax(q / T, dim=1).detach()
    return loss_kl(soft_p, soft_q)


def propagate(features, k, adj_norm):
    feature_list = []
    feature_list.append(features)
    for i in range(k):
        feature_list.append(torch.spmm(adj_norm, feature_list[-1]))
    return feature_list[-1]


def get_adj_mat(coo_mat, num_nodes):
    mat = torch.zeros(num_nodes, num_nodes)

    for i in range(len(coo_mat[0])):
        a, b = coo_mat[0][i].item(), coo_mat[1][i].item()
        mat[a][b] = 1
    mat = torch.tensor(mat, dtype=torch.float32)
    return mat


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def calc_f1(y_true, y_pred, mask, multilabel=False):
    if multilabel:
        y_pred[y_pred > 0.5] = 1
        y_pred[y_pred <= 0.5] = 0
    else:
        y_pred = np.argmax(y_pred, axis=1)
    mask = mask.cpu()
    return f1_score(y_true[mask], y_pred[mask], average="micro")


def normalize_adj(adj, r=0.5):
    adj = adj + sp.eye(adj.shape[0])
    degrees = np.array(adj.sum(1))
    r_inv_sqrt_left = np.power(degrees, r - 1).flatten()
    r_inv_sqrt_left[np.isinf(r_inv_sqrt_left)] = 0.0
    r_mat_inv_sqrt_left = sp.diags(r_inv_sqrt_left)

    r_inv_sqrt_right = np.power(degrees, -r).flatten()
    r_inv_sqrt_right[np.isinf(r_inv_sqrt_right)] = 0.0
    r_mat_inv_sqrt_right = sp.diags(r_inv_sqrt_right)

    adj_normalized = adj.dot(r_mat_inv_sqrt_left).transpose().dot(r_mat_inv_sqrt_right)
    return adj_normalized


class MEGU(UnlearnMethod):
    def __init__(self, model, data):
        self.device = "cpu"
        self.args = args
        self.target_model = model
        self.data = data
        self.num_layers = 2

    def unlearn(self):
        self.train_test_split()
        self.unlearning_request()
        self.adj = sparse_mx_to_torch_sparse_tensor(
            normalize_adj(to_scipy_sparse_matrix(self.data.edge_index))
        )
        self.neighbor_khop = self.neighbor_select(self.data.x)

        run_f1 = np.empty(0)
        run_f1_unlearning = np.empty(0)
        unlearning_times = np.empty(0)
        training_times = np.empty(0)

        for run in range(self.args.megu_num_runs):
            f1_score = self.evaluate(run)
            run_f1 = np.append(run_f1, f1_score)

            # unlearning with MEGU
            unlearning_time, f1_score_unlearning = self.megu_training()
            unlearning_times = np.append(unlearning_times, unlearning_time)
            run_f1_unlearning = np.append(run_f1_unlearning, f1_score_unlearning)

        f1_score_unlearning_avg = str(np.average(run_f1_unlearning)).split(".")[1]
        f1_score_unlearning_std = str(np.std(run_f1_unlearning)).split(".")[1]
        unlearning_time_avg = np.average(unlearning_times)

        f1_score_unlearning_avg = ".".join(
            (f1_score_unlearning_avg[0:2], f1_score_unlearning_avg[2:4])
        )
        f1_score_unlearning_std = ".".join(
            (f1_score_unlearning_std[1:2], f1_score_unlearning_std[2:4])
        )
        print(
            f"|Unlearn| f1_score: avg±std={f1_score_unlearning_avg}±{f1_score_unlearning_std} time: avg={np.average(unlearning_times):.4f}s"
        )
        return self.target_model

    def train_test_split(self):
        print(self.args)
        self.train_indices, self.test_indices = train_test_split(
            np.arange(self.data.num_nodes),
            test_size=self.args.megu_test_ratio,
            random_state=100,
        )

        self.data.train_mask = torch.from_numpy(
            np.isin(np.arange(self.data.num_nodes), self.train_indices)
        )
        self.data.test_mask = torch.from_numpy(
            np.isin(np.arange(self.data.num_nodes), self.test_indices)
        )

    def unlearning_request(self):
        # self.logger.debug("Train data  #.Nodes: %f, #.Edges: %f" % (
        #     self.data.num_nodes, self.data.num_edges))

        self.data.x_unlearn = self.data.x.clone()
        self.data.edge_index_unlearn = self.data.edge_index.clone()
        edge_index = self.data.edge_index.numpy()
        unique_indices = np.where(edge_index[0] < edge_index[1])[0]

        if self.unlearn_request == "node":
            # unique_nodes = np.random.choice(
            #     len(self.train_indices),
            #     int(len(self.train_indices) * self.args["unlearn_ratio"]),
            #     replace=False,
            # )
            unique_nodes= self.nodes_to_unlearn
            self.data.edge_index_unlearn = self.update_edge_index_unlearn(unique_nodes)

        # if self.args["unlearn_task"] == "edge":
        #     remove_indices = np.random.choice(
        #         unique_indices,
        #         int(unique_indices.shape[0] * self.args["unlearn_ratio"]),
        #         replace=False,
        #     )
        #     remove_edges = edge_index[:, remove_indices]
        #     unique_nodes = np.unique(remove_edges)

        #     self.data.edge_index_unlearn = self.update_edge_index_unlearn(
        #         unique_nodes, remove_indices
        #     )

        # if self.args["unlearn_task"] == "feature":
        #     unique_nodes = np.random.choice(
        #         len(self.train_indices),
        #         int(len(self.train_indices) * self.args["unlearn_ratio"]),
        #         replace=False,
        #     )
        #     self.data.x_unlearn[unique_nodes] = 0.0

        self.temp_node = unique_nodes

    def update_edge_index_unlearn(self, delete_nodes, delete_edge_index=None):
        edge_index = self.data.edge_index.numpy()

        unique_indices = np.where(edge_index[0] < edge_index[1])[0]
        unique_indices_not = np.where(edge_index[0] > edge_index[1])[0]

        if self.unlearn_request == "edge":
            remain_indices = np.setdiff1d(unique_indices, delete_edge_index)
        else:
            unique_edge_index = edge_index[:, unique_indices]
            delete_edge_indices = np.logical_or(
                np.isin(unique_edge_index[0], delete_nodes),
                np.isin(unique_edge_index[1], delete_nodes),
            )
            remain_indices = np.logical_not(delete_edge_indices)
            remain_indices = np.where(remain_indices == True)[0]

        remain_encode = (
            edge_index[0, remain_indices] * edge_index.shape[1] * 2
            + edge_index[1, remain_indices]
        )
        unique_encode_not = (
            edge_index[1, unique_indices_not] * edge_index.shape[1] * 2
            + edge_index[0, unique_indices_not]
        )
        sort_indices = np.argsort(unique_encode_not)

        print(f"sort_indices: {sort_indices}")
        print(f"unique_encode_not: {unique_encode_not}")
        print(f"remain_encode: {remain_encode}")

        indices_to_check = np.searchsorted(
            unique_encode_not, remain_encode, sorter=sort_indices
        )
        valid_indices = indices_to_check[indices_to_check < unique_encode_not.size]
        valid_remain_encode = remain_encode[indices_to_check < unique_encode_not.size]

        remain_indices_not = unique_indices_not[
            sort_indices[
                np.searchsorted(
                    unique_encode_not, valid_remain_encode, sorter=sort_indices
                )
            ]
        ]
        remain_indices = np.union1d(remain_indices, remain_indices_not)

        return torch.from_numpy(edge_index[:, remain_indices])

    def evaluate(self, run):
        # self.logger.info('model evaluation')

        start_time = time.time()
        self.target_model.eval()
        out = self.target_model(
            self.data.x, get_adj_mat(self.data.edge_index, self.data.num_nodes)
        )
        y = self.data.y.cpu()
        if self.args.dataset_name == "ppi":
            y_hat = torch.sigmoid(out).cpu().detach().numpy()
            test_f1 = calc_f1(y, y_hat, self.data.test_mask, multilabel=True)
        else:
            y_hat = F.log_softmax(out, dim=1).cpu().detach().numpy()
            test_f1 = calc_f1(y, y_hat, self.data.test_mask)

        evaluate_time = time.time() - start_time
        # self.logger.info(f"Evaluation cost {evaluate_time:.4f} seconds.")

        # self.logger.info(f"Final Test F1: {test_f1:.4f}")
        return test_f1

    def neighbor_select(self, features):
        temp_features = features.clone()
        pfeatures = propagate(temp_features, self.num_layers, self.adj)
        reverse_feature = self.reverse_features(temp_features)
        re_pfeatures = propagate(reverse_feature, self.num_layers, self.adj)

        cos = nn.CosineSimilarity()
        sim = cos(pfeatures, re_pfeatures)

        alpha = 0.1
        gamma = 0.1
        max_val = 0.0
        while True:
            influence_nodes_with_unlearning_nodes = (
                torch.nonzero(sim <= alpha).flatten().cpu()
            )
            if len(influence_nodes_with_unlearning_nodes.view(-1)) > 0:
                temp_max = torch.max(sim[influence_nodes_with_unlearning_nodes])
            else:
                alpha = alpha + gamma
                continue

            if temp_max == max_val:
                break

            max_val = temp_max
            alpha = alpha + gamma

        # influence_nodes_with_unlearning_nodes = torch.nonzero(sim < 0.5).squeeze().cpu()
        neighborkhop, _, _, two_hop_mask = k_hop_subgraph(
            torch.tensor(self.temp_node),
            self.num_layers,
            self.data.edge_index,
            num_nodes=self.data.num_nodes,
        )

        neighborkhop = neighborkhop[~np.isin(neighborkhop.cpu(), self.temp_node)]
        neighbor_nodes = []
        for idx in influence_nodes_with_unlearning_nodes:
            if idx in neighborkhop and idx not in self.temp_node:
                neighbor_nodes.append(idx.item())

        neighbor_nodes_mask = torch.from_numpy(
            np.isin(np.arange(self.data.num_nodes), neighbor_nodes)
        )

        return neighbor_nodes_mask

    def reverse_features(self, features):
        reverse_features = features.clone()
        for idx in self.temp_node:
            reverse_features[idx] = 1 - reverse_features[idx]

        return reverse_features

    def correct_and_smooth(self, y_soft, preds):
        pos = CorrectAndSmooth(
            num_correction_layers=80,
            correction_alpha=self.args.alpha1,
            num_smoothing_layers=80,
            smoothing_alpha=self.args.alpha2,
            autoscale=False,
            scale=1.0,
        )

        y_soft = pos.correct(
            y_soft,
            preds[self.data.train_mask].type(torch.LongTensor),
            self.data.train_mask,
            self.data.edge_index_unlearn.type(torch.LongTensor),
        )
        y_soft = pos.smooth(
            y_soft,
            preds[self.data.train_mask].type(torch.LongTensor),
            self.data.train_mask,
            self.data.edge_index_unlearn.type(torch.LongTensor),
        )

        return y_soft

    def megu_training(self):
        operator = GATE(self.data.num_classes).to(self.device)

        optimizer = torch.optim.SGD(
            [
                {"params": self.target_model.parameters()},
                {"params": operator.parameters()},
            ],
            lr=self.args.megu_unlearn_lr,
        )

        with torch.no_grad():
            self.target_model.eval()
            preds = self.target_model(
                self.data.x, get_adj_mat(self.data.edge_index, self.data.num_nodes)
            )
            if self.args.dataset_name == "ppi":
                preds = torch.sigmoid(preds).ge(0.5)
                preds = preds.type_as(self.data.y)
            else:
                preds = torch.argmax(preds, axis=1).type_as(self.data.y)

        start_time = time.time()
        for epoch in range(self.args.megu_num_epochs):
            self.target_model.train()
            operator.train()
            optimizer.zero_grad()

            # print("SHAPEEEE ERORORROROR")
            # print(self.data.x_unlearn.shape)
            # print(get_adj_mat(self.data.edge_index_unlearn, self.data.num_nodes).shape)

            out_ori = self.target_model(
                self.data.x_unlearn,
                get_adj_mat(self.data.edge_index_unlearn, self.data.num_nodes),
            )
            out = operator(out_ori)

            if self.args.dataset_name == "ppi":
                loss_u = criterionKD(
                    out_ori[self.temp_node], out[self.temp_node]
                ) - F.binary_cross_entropy_with_logits(
                    out[self.temp_node], preds[self.temp_node]
                )
                loss_r = criterionKD(
                    out[self.neighbor_khop], out_ori[self.neighbor_khop]
                ) + F.binary_cross_entropy_with_logits(
                    out_ori[self.neighbor_khop], preds[self.neighbor_khop]
                )
            else:
                loss_u = criterionKD(
                    out_ori[self.temp_node], out[self.temp_node]
                ) - F.cross_entropy(
                    out[self.temp_node], preds[self.temp_node].type(torch.LongTensor)
                )

                loss_r = criterionKD(
                    out[self.neighbor_khop], out_ori[self.neighbor_khop]
                ) + F.cross_entropy(
                    out_ori[self.neighbor_khop],
                    preds[self.neighbor_khop].type(torch.LongTensor),
                )

            loss = self.args.kappa * loss_u + loss_r
            # print(f"Val Loss: {loss}")
            loss.backward()
            optimizer.step()

        unlearn_time = time.time() - start_time
        self.target_model.eval()
        test_out = self.target_model(
            self.data.x_unlearn,
            get_adj_mat(self.data.edge_index_unlearn, self.data.num_nodes),
        )
        if self.args.dataset_name == "ppi":
            out = torch.sigmoid(test_out)
        else:
            out = self.correct_and_smooth(F.softmax(test_out, dim=-1), preds)

        y_hat = out.cpu().detach().numpy()
        y = self.data.y.cpu()
        if self.args.dataset_name == "ppi":
            test_f1 = calc_f1(y, y_hat, self.data.test_mask, multilabel=True)
        else:
            print(self.data.test_mask)
            print(self.data.test_mask.shape)
            print(sum(self.data.test_mask))
            test_f1 = calc_f1(y, y_hat, self.data.test_mask)

        return unlearn_time, test_f1
