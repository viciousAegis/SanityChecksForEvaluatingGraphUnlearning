import torch
import torch.nn as nn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.autograd.set_detect_anomaly(True)
import time
from torch_geometric.utils import negative_sampling, k_hop_subgraph
import torch.nn.functional as F
import math
from gub.config import args

#Losses: Hyperparameter
def to_directed(edge_index):
    row, col = edge_index
    mask = row < col
    return torch.cat([row[mask], col[mask]], dim=0)
def BoundedKLDMean(logits, truth):
    return 1 - torch.exp(-F.kl_div(F.log_softmax(logits, -1), truth.softmax(-1), None, None, 'batchmean'))
def BoundedKLDSum(logits, truth):
    return 1 - torch.exp(-F.kl_div(F.log_softmax(logits, -1), truth.softmax(-1), None, None, 'sum'))
def CosineDistanceMean(logits, truth):
    return (1 - F.cosine_similarity(logits, truth)).mean()
def CosineDistanceSum(logits, truth):
    return (1 - F.cosine_similarity(logits, truth)).sum()
def centering(K):
    n = K.shape[0]
    unit = torch.ones([n, n], device=K.device)
    I = torch.eye(n, device=K.device)
    H = I - unit / n
    return torch.matmul(torch.matmul(H, K), H)
def rbf(X, sigma=None):
    GX = torch.matmul(X, X.T)
    KX = torch.diag(GX) - GX + (torch.diag(GX) - GX).T
    if sigma is None:
        mdist = torch.median(KX[KX != 0])
        sigma = math.sqrt(mdist)
    KX *= - 0.5 / (sigma * sigma)
    KX = torch.exp(KX)
    return KX
def kernel_HSIC(X, Y, sigma=None):
    return torch.sum(centering(rbf(X, sigma)) * centering(rbf(Y, sigma)))
def linear_HSIC(X, Y):
    L_X = torch.matmul(X, X.T)
    L_Y = torch.matmul(Y, Y.T)
    return torch.sum(centering(L_X) * centering(L_Y))
def LinearCKA(X, Y):
    hsic = linear_HSIC(X, Y)
    var1 = torch.sqrt(linear_HSIC(X, X))
    var2 = torch.sqrt(linear_HSIC(Y, Y))
    return hsic / (var1 * var2)
def RBFCKA(X, Y, sigma=None):
    hsic = kernel_HSIC(X, Y, sigma)
    var1 = torch.sqrt(kernel_HSIC(X, X, sigma))
    var2 = torch.sqrt(kernel_HSIC(Y, Y, sigma))
    return hsic / (var1 * var2)
def get_loss_fct(name):
    if name == 'kld_mean':
        loss_fct = BoundedKLDMean
    elif name == 'kld_sum':
        loss_fct = BoundedKLDSum
    elif name == 'mse_mean':
        loss_fct = nn.MSELoss(reduction='mean')
    elif name == 'mse_sum':
        loss_fct = nn.MSELoss(reduction='sum')
    elif name == 'cosine_mean':
        loss_fct = CosineDistanceMean
    elif name == 'cosine_sum':
        loss_fct = CosineDistanceSum
    elif name == 'linear_cka':
        loss_fct = LinearCKA
    elif name == 'rbf_cka':
        loss_fct = RBFCKA
    else:
        raise NotImplementedError
    return loss_fct


class GNNDeleteNodeClassificationTrainer:
    def __init__(self, args):
        self.args= args

    def train(self, model, data, optimizer):
        model = model.to('cuda')
        data = data.to('cuda')
        non_df_node_mask = torch.ones(data.x.shape[0], dtype=torch.bool, device=data.x.device)
        non_df_node_mask[data.directed_df_edge_index.flatten().unique()] = False

        data.sdf_node_1hop_mask_non_df_mask = data.sdf_node_1hop_mask & non_df_node_mask
        data.sdf_node_2hop_mask_non_df_mask = data.sdf_node_2hop_mask & non_df_node_mask

        with torch.no_grad():
            z1_ori, z2_ori = model.get_original_embeddings(data.x, data.edge_index[:, data.dr_mask], return_all_emb=True)

        loss_fct = get_loss_fct(self.args.loss_fct)

        neg_edge = neg_edge_index = negative_sampling(
            edge_index=data.edge_index,
            num_nodes=data.num_nodes,
            num_neg_samples=data.df_mask.sum())

        for _ in self.args.gnndelete_epochs:
            model.train()

            start_time = time.time()
            z1, z2 = model(data.x, data.edge_index[:, data.sdf_mask], return_all_emb=True)
            pos_edge = data.edge_index[:, data.df_mask]

            embed1 = torch.cat([z1[pos_edge[0]], z1[pos_edge[1]]], dim=0)
            embed1_ori = torch.cat([z1_ori[neg_edge[0]], z1_ori[neg_edge[1]]], dim=0)
            embed2 = torch.cat([z2[pos_edge[0]], z2[pos_edge[1]]], dim=0)
            embed2_ori = torch.cat([z2_ori[neg_edge[0]], z2_ori[neg_edge[1]]], dim=0)

            loss_r1 = loss_fct(embed1, embed1_ori)
            loss_r2 = loss_fct(embed2, embed2_ori)
            loss_l1 = loss_fct(z1[data.sdf_node_1hop_mask_non_df_mask], z1_ori[data.sdf_node_1hop_mask_non_df_mask])
            loss_l2 = loss_fct(z2[data.sdf_node_2hop_mask_non_df_mask], z2_ori[data.sdf_node_2hop_mask_non_df_mask])

            # loss_l = loss_l1 + loss_l2
            # loss_r = loss_r1 + loss_r2

            loss1 = self.args.alpha * loss_r1 + (1 - self.args.alpha) * loss_l1
            loss1.backward(retain_graph=True)
            optimizer[0].step()
            optimizer[0].zero_grad()

            loss2 = self.args.alpha * loss_r2 + (1 - self.args.alpha) * loss_l2
            loss2.backward(retain_graph=True)
            optimizer[1].step()
            optimizer[1].zero_grad()

            loss = loss1 + loss2
            end_time = time.time()
            epoch_time = end_time - start_time

class GNNDelete:
    def __init__(self, model, data):
        self.model= model
        self.data= data
        self.args= args
        self.unlearn()

    def unlearn(self):
        if self.args.df_size >= 100:
            df_size = int(self.args.df_size)
        else:
            df_size = int(self.args.df_size / 100 * self.data.train_pos_edge_index.shape[1])
        print(f'Original size: {self.data.num_nodes:,}')
        print(f'Df size: {df_size:,}')

        df_nodes = self.data.deletion_indices
        df_size= len(df_nodes)
        global_node_mask = torch.ones(self.data.num_nodes, dtype=torch.bool)
        global_node_mask[df_nodes] = False

        dr_mask_node = global_node_mask
        df_mask_node = ~global_node_mask
        assert df_mask_node.sum() == df_size

        res = [torch.eq(self.data.edge_index, aelem).logical_or_(torch.eq(self.data.edge_index, aelem)) for aelem in df_nodes]
        df_mask_edge = torch.any(torch.stack(res, dim=0), dim = 0)
        df_mask_edge = df_mask_edge.sum(0).bool()
        dr_mask_edge = ~df_mask_edge

        df_edge = self.data.edge_index[:, df_mask_edge]
        self.data.directed_df_edge_index = to_directed(df_edge)

        _, two_hop_edge, _, two_hop_mask = k_hop_subgraph(
            self.data.edge_index[:, df_mask_edge].flatten().unique(),
            2,
            self.data.edge_index,
            num_nodes=self.data.num_nodes)

        _, one_hop_edge, _, one_hop_mask = k_hop_subgraph(
            self.data.edge_index[:, df_mask_edge].flatten().unique(),
            1,
            self.data.edge_index,
            num_nodes=self.data.num_nodes)
        sdf_node_1hop = torch.zeros(self.data.num_nodes, dtype=torch.bool)
        sdf_node_2hop = torch.zeros(self.data.num_nodes, dtype=torch.bool)

        sdf_node_1hop[one_hop_edge.flatten().unique()] = True
        sdf_node_2hop[two_hop_edge.flatten().unique()] = True

        assert sdf_node_1hop.sum() == len(one_hop_edge.flatten().unique())
        assert sdf_node_2hop.sum() == len(two_hop_edge.flatten().unique())

        self.data.sdf_node_1hop_mask = sdf_node_1hop
        self.data.sdf_node_2hop_mask = sdf_node_2hop


        two_hop_mask = two_hop_mask.bool()
        df_mask_edge = df_mask_edge.bool()
        dr_mask_edge = ~df_mask_edge

        self.data.sdf_mask = two_hop_mask
        self.data.df_mask = df_mask_edge
        self.data.dr_mask = dr_mask_edge
        self.data.dtrain_mask = dr_mask_edge
        model = model.to(device)

        if 'nodeemb' in self.args.unlearning_model:
            parameters_to_optimize = [
                {'params': [p for n, p in model.named_parameters() if 'del' in n], 'weight_decay': 0.0}
            ]
            print('parameters_to_optimize', [n for n, p in model.named_parameters() if 'del' in n])
            if 'layerwise' in self.args.loss_type:
                optimizer1 = torch.optim.Adam(model.deletion1.parameters(), lr=self.args.gnndelete_lr)
                optimizer2 = torch.optim.Adam(model.deletion2.parameters(), lr=self.args.gnndelete_lr)
                optimizer = [optimizer1, optimizer2]
            else:
                optimizer = torch.optim.Adam(parameters_to_optimize, lr=self.args.gnndelete_lr)
        else:
            parameters_to_optimize = [
                {'params': [p for n, p in model.named_parameters() if 'del' in n], 'weight_decay': 0.0}
            ]
            print('parameters_to_optimize', [n for n, p in model.named_parameters() if 'del' in n])
            optimizer = torch.optim.Adam(parameters_to_optimize, lr=self.args.gnndelete_lr)#, weight_decay=self.args.weight_decay)

        trainer = GNNDeleteNodeClassificationTrainer(self.args)
        trainer.train(model, self.data, optimizer, self.args)
