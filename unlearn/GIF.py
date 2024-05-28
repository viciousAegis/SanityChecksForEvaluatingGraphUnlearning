from unlearn.UnlearnMethod import UnlearnMethod

from cgi import test
import logging
import time
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

torch.cuda.empty_cache()
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.data import NeighborSampler
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import numpy as np
from src.model_utils import test_model
from src.config import args

class ExpGraphInfluenceFunction(UnlearnMethod):
    def __init__(self, model, data):
        self.deleted_nodes = np.array([])
        self.feature_nodes = np.array([])
        self.influence_nodes = np.array([])

        self.target_model= model
        self.args=args
        self.data=data

    def unlearn(self):
        # self.train_test_split() (Use default train and test mask of data)
        self.unlearning_request()

        run_f1 = np.empty((0))
        run_f1_unlearning = np.empty((0))
        unlearning_times = np.empty((0))
        training_times = np.empty((0))

        # self.best_model= self.target_model
        # best_score= test_model(self.target_model, self.data)

        for run in range(self.args.gif_num_runs):
            run_training_time, result_tuple = self._train_model(run)
            # f1_score = self.evaluate(run)
            run_f1 = np.append(run_f1, f1_score)
            training_times = np.append(training_times, run_training_time)

            # unlearning with GIF
            unlearning_time, f1_score_unlearning = self.gif_approxi(result_tuple)
            unlearning_times = np.append(unlearning_times, unlearning_time)
            run_f1_unlearning = np.append(run_f1_unlearning, f1_score_unlearning)

            #saving best model
            # test_score= test_model(self.target_model, self.data)
            # if(test_score>=best_score):
            #     best_score= test_score
            #     self.best_model= self.target_model

        return self.target_model

        # f1_score_avg = np.average(run_f1)
        # f1_score_std = np.std(run_f1)

        # f1_score_unlearning_avg = np.average(run_f1_unlearning)
        # f1_score_unlearning_std = np.std(run_f1_unlearning)
        # unlearning_time_avg = np.average(unlearning_times)
        # print(f1_score_avg, f1_score_std, f1_score_unlearning_avg, f1_score_unlearning_std, unlearning_time_avg)


    def train_test_split(self):
        self.train_indices, self.test_indices = train_test_split(np.arange((self.data.num_nodes)), test_size=self.args.gif_test_ratio, random_state=100)
        self.data.train_mask = torch.from_numpy(np.isin(np.arange(self.data.num_nodes), self.train_indices))
        self.data.test_mask = torch.from_numpy(np.isin(np.arange(self.data.num_nodes), self.test_indices))


    def unlearning_request(self):
        self.data.x_unlearn = self.data.x.clone()
        self.data.edge_index_unlearn = self.data.edge_index.clone()
        # unique_nodes = np.random.choice(len(self.train_indices),
        #                                 int(len(self.train_indices) * self.args['unlearn_ratio']),
        #                                 replace=False)
        unique_nodes= self.data.deletion_indices
        self.data.edge_index_unlearn = self.update_edge_index_unlearn(unique_nodes)
        self.find_k_hops(unique_nodes)


    def update_edge_index_unlearn(self, delete_nodes, delete_edge_index=None):
        edge_index = self.data.edge_index.numpy()

        unique_indices = np.where(edge_index[0] < edge_index[1])[0]
        unique_indices_not = np.where(edge_index[0] > edge_index[1])[0]

        unique_edge_index = edge_index[:, unique_indices]
        delete_edge_indices = np.logical_or(np.isin(unique_edge_index[0], delete_nodes),
                                            np.isin(unique_edge_index[1], delete_nodes))
        remain_indices = np.logical_not(delete_edge_indices)
        remain_indices = np.where(remain_indices == True)[0]

        remain_encode = edge_index[0, remain_indices] * edge_index.shape[1] * 2 + edge_index[1, remain_indices]
        unique_encode_not = edge_index[1, unique_indices_not] * edge_index.shape[1] * 2 + edge_index[0, unique_indices_not]
        sort_indices = np.argsort(unique_encode_not)

        print(f"sort_indices: {sort_indices}")
        print(f"unique_encode_not: {unique_encode_not}")
        print(f"remain_encode: {remain_encode}")

        indices_to_check = np.searchsorted(unique_encode_not, remain_encode, sorter=sort_indices)
        valid_indices = indices_to_check[indices_to_check < unique_encode_not.size]
        valid_remain_encode = remain_encode[indices_to_check < unique_encode_not.size]

        remain_indices_not = unique_indices_not[
            sort_indices[np.searchsorted(unique_encode_not, valid_remain_encode, sorter=sort_indices)]]
        remain_indices = np.union1d(remain_indices, remain_indices_not)

        return torch.from_numpy(edge_index[:, remain_indices])

    #f1 doesnt work well
    # def evaluate(self, run):
    #     posterior = self.target_model.posterior()
    #     test_f1 = f1_score(
    #         self.data.y[self.data['test_mask']].cpu().numpy(),
    #         posterior.argmax(axis=1).cpu().numpy(),
    #         average="micro"
    #     )
    #     return test_f1

    def get_adj_mat(self, coo_mat, num_nodes):
        mat = torch.zeros(num_nodes, num_nodes)

        for i in range(len(coo_mat[0])):
            a, b = coo_mat[0][i].item(), coo_mat[1][i].item()
            mat[a][b] = 1
        mat = torch.tensor(mat, dtype=torch.float32)
        return mat

    def get_gradients(self, unlearn_info=None):
        grad_all, grad1, grad2 = None, None, None

        out1= self.target_model.forward(self.data.x, self.get_adj_mat(self.data.edge_index, self.data.num_nodes))
        out2= self.target_model.forward(self.data.x_unlearn, self.get_adj_mat(self.data.edge_index, self.data.num_nodes))

        mask1 = np.array([False] * out1.shape[0])
        mask1[unlearn_info[0]] = True
        mask1[unlearn_info[2]] = True
        mask2 = np.array([False] * out2.shape[0])
        mask2[unlearn_info[2]] = True

        loss = F.nll_loss(out1[self.data.train_mask], self.data.y[self.data.train_mask].type(torch.LongTensor), reduction='sum')
        loss1 = F.nll_loss(out1[mask1], self.data.y[mask1].type(torch.LongTensor), reduction='sum')
        loss2 = F.nll_loss(out2[mask2], self.data.y[mask2].type(torch.LongTensor), reduction='sum')
        model_params = [p for p in self.target_model.parameters() if p.requires_grad]
        grad_all = grad(loss, model_params, retain_graph=True, create_graph=True)
        grad1 = grad(loss1, model_params, retain_graph=True, create_graph=True)
        grad2 = grad(loss2, model_params, retain_graph=True, create_graph=True)
        return (grad_all, grad1, grad2)

    def _train_model(self, run):
        start_time = time.time()
        self.target_model.data = self.data
        res = self.get_gradients(
            (self.deleted_nodes, self.feature_nodes, self.influence_nodes))
        train_time = time.time() - start_time
        return train_time, res


    def find_k_hops(self, unique_nodes):
        edge_index = self.data.edge_index.numpy()
        hops = 3
        influenced_nodes = unique_nodes
        for _ in range(hops):
            target_nodes_location = np.isin(edge_index[0], influenced_nodes)
            neighbor_nodes = edge_index[1, target_nodes_location]
            influenced_nodes = np.append(influenced_nodes, neighbor_nodes)
            influenced_nodes = np.unique(influenced_nodes)

        neighbor_nodes = np.setdiff1d(influenced_nodes, unique_nodes)
        self.deleted_nodes = unique_nodes
        self.influence_nodes = neighbor_nodes


    def gif_approxi(self, res_tuple):
        start_time = time.time()
        iteration, damp, scale = self.args.iteration, self.args.damp, self.args.scale
        v = tuple(grad1 - grad2 for grad1, grad2 in zip(res_tuple[1], res_tuple[2]))
        h_estimate = tuple(grad1 - grad2 for grad1, grad2 in zip(res_tuple[1], res_tuple[2]))
        for _ in range(iteration):
            model_params  = [p for p in self.target_model.parameters() if p.requires_grad]
            hv            = self.hvps(res_tuple[0], model_params, h_estimate)
            with torch.no_grad():
                h_estimate    = [ v1 + (1-damp)*h_estimate1 - hv1/scale
                            for v1, h_estimate1, hv1 in zip(v, h_estimate, hv)]
        params_change = [h_est / scale for h_est in h_estimate]
        params_esti   = [p1 + p2 for p1, p2 in zip(params_change, model_params)]
        # test_F1 = self.target_model.evaluate_unlearn_F1(params_esti)
        test_F1=0
        return time.time() - start_time, test_F1


    def hvps(self, grad_all, model_params, h_estimate):
        element_product = 0
        for grad_elem, v_elem in zip(grad_all, h_estimate):
            element_product += torch.sum(grad_elem * v_elem)
        return_grads = grad(element_product,model_params,create_graph=True)
        return return_grads