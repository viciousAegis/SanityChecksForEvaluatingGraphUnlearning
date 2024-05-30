import os
import math
import time
import wandb

from tqdm import trange, tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import k_hop_subgraph, is_undirected
import torch_geometric.transforms as T
from torch_geometric.seed import seed_everything
from torch_geometric.utils import negative_sampling
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score

from grb.model.torch.gcn import GCN
from grb.model.torch.gin import GIN
from gub.config import args
from grb.trainer.trainer import Trainer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.autograd.set_detect_anomaly(True)

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
    # if name == 'mse':
    #     loss_fct = nn.MSELoss(reduction='mean')
    # elif name == 'kld':
    #     loss_fct = BoundedKLDMean
    # elif name == 'cosine':
    #     loss_fct = CosineDistanceMean

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

class NodeClassificationTrainer(Trainer):
    def train(self, model, data, optimizer, args):
        start_time = time.time()
        best_epoch = 0
        best_valid_acc = 0

        data = data.to(device)
        for epoch in trange(args['epochs'], desc='Epoch'):
            model.train()

            z = F.log_softmax(model(data.x, data.edge_index), dim=1)
            loss = F.nll_loss(z[data.train_mask], data.y[data.train_mask])

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if (epoch+1) % args['valid_freq'] == 0:
                valid_loss, dt_acc, dt_f1, valid_log = self.eval(model, data, 'val')

                train_log = {
                    'epoch': epoch,
                    'train_loss': loss.item()
                }

                for log in [train_log, valid_log]:
                    wandb.log(log)
                    msg = [f'{i}: {j:>4d}' if isinstance(j, int) else f'{i}: {j:.4f}' for i, j in log.items()]
                    tqdm.write(' | '.join(msg))

                self.trainer_log['log'].append(train_log)
                self.trainer_log['log'].append(valid_log)

                if dt_acc > best_valid_acc:
                    best_valid_acc = dt_acc
                    best_epoch = epoch

                    print(f'Save best checkpoint at epoch {epoch:04d}. Valid Acc = {dt_acc:.4f}')
                    ckpt = {
                        'model_state': model.state_dict(),
                        'optimizer_state': optimizer.state_dict(),
                    }
                    torch.save(ckpt, os.path.join(args['checkpoint_dir'], 'model_best.pt'))
                    torch.save(z, os.path.join(args['checkpoint_dir'], 'node_embeddings.pt'))

        self.trainer_log['training_time'] = time.time() - start_time

        # Save models and node embeddings
        print('Saving final checkpoint')
        ckpt = {
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
        }
        torch.save(ckpt, os.path.join(args['checkpoint_dir'], 'model_final.pt'))

        print(f'Training finished. Best checkpoint at epoch = {best_epoch:04d}, best valid acc = {best_valid_acc:.4f}')

        self.trainer_log['best_epoch'] = best_epoch
        self.trainer_log['best_valid_acc'] = best_valid_acc

    @torch.no_grad()
    def eval(self, model, data, stage='val', pred_all=False):
        model.eval()

        if self.args['eval_on_cpu']:
            model = model.to('cpu')

        # if hasattr(data, 'dtrain_mask'):
        #     mask = data.dtrain_mask
        # else:
        #     mask = data.dr_mask
        z = F.log_softmax(model(data.x, data.edge_index), dim=1)

        # DT AUC AUP
        loss = F.nll_loss(z[data.val_mask], data.y[data.val_mask]).cpu().item()
        pred = torch.argmax(z[data.val_mask], dim=1).cpu()
        dt_acc = accuracy_score(data.y[data.val_mask].cpu(), pred)
        dt_f1 = f1_score(data.y[data.val_mask].cpu(), pred, average='micro')

        # DF AUC AUP
        # if self.args['unlearning_model in ['original', 'original_node']:
        #     df_logit = []
        # else:
        #     df_logit = model.decode(z, data.directed_df_edge_index).sigmoid().tolist()

        # if len(df_logit) > 0:
        #     df_auc = []
        #     df_aup = []

        #     # Sample pos samples
        #     if len(self.df_pos_edge) == 0:
        #         for i in range(500):
        #             mask = torch.zeros(data.train_pos_edge_index[:, data.dr_mask].shape[1], dtype=torch.bool)
        #             idx = torch.randperm(data.train_pos_edge_index[:, data.dr_mask].shape[1])[:len(df_logit)]
        #             mask[idx] = True
        #             self.df_pos_edge.append(mask)

        #     # Use cached pos samples
        #     for mask in self.df_pos_edge:
        #         pos_logit = model.decode(z, data.train_pos_edge_index[:, data.dr_mask][:, mask]).sigmoid().tolist()

        #         logit = df_logit + pos_logit
        #         label = [0] * len(df_logit) +  [1] * len(df_logit)
        #         df_auc.append(roc_auc_score(label, logit))
        #         df_aup.append(average_precision_score(label, logit))

        #     df_auc = np.mean(df_auc)
        #     df_aup = np.mean(df_aup)

        # else:
        #     df_auc = np.nan
        #     df_aup = np.nan

        # Logits for all node pairs
        if pred_all:
            logit_all_pair = (z @ z.t()).cpu()
        else:
            logit_all_pair = None

        log = {
            f'{stage}_loss': loss,
            f'{stage}_dt_acc': dt_acc,
            f'{stage}_dt_f1': dt_f1,
        }

        if self.args['eval_on_cpu']:
            model = model.to(device)

        return loss, dt_acc, dt_f1, log

    @torch.no_grad()
    def test(self, model, data, model_retrain=None, attack_model_all=None, attack_model_sub=None, ckpt='best'):

        if ckpt == 'best':    # Load best ckpt
            ckpt = torch.load(os.path.join(self.args['checkpoint_dir'], 'model_best.pt'))
            model.load_state_dict(ckpt['model_state'])

        if 'ogbl' in 'cora':
            pred_all = False
        else:
            pred_all = True
        loss, dt_acc, dt_f1, test_log = self.eval(model, data, 'test', pred_all)

        self.trainer_log['dt_loss'] = loss
        self.trainer_log['dt_acc'] = dt_acc
        self.trainer_log['dt_f1'] = dt_f1
        # self.trainer_log['df_logit'] = df_logit
        # self.logit_all_pair = logit_all_pair
        # self.trainer_log['df_auc'] = df_auc
        # self.trainer_log['df_aup'] = df_aup

        return loss, dt_acc, dt_f1, test_log

class GNNDeleteNodeClassificationTrainer(NodeClassificationTrainer):
    def train(self, model, data, optimizer, args, logits_ori=None, attack_model_all=None, attack_model_sub=None):
        model = model.to('cuda')
        data = data.to('cuda')

        best_metric = 0

        non_df_node_mask = torch.ones(data.x.shape[0], dtype=torch.bool, device=data.x.device)
        non_df_node_mask[data.directed_df_edge_index.flatten().unique()] = False

        data.sdf_node_1hop_mask_non_df_mask = data.sdf_node_1hop_mask & non_df_node_mask
        data.sdf_node_2hop_mask_non_df_mask = data.sdf_node_2hop_mask & non_df_node_mask

        # Original node embeddings
        with torch.no_grad():
            z1_ori, z2_ori = model.get_original_embeddings(data.x, data.edge_index[:, data.dr_mask], return_all_emb=True)

        loss_fct = get_loss_fct(self.args['loss_fct'])

        neg_edge = neg_edge_index = negative_sampling(
            edge_index=data.edge_index,
            num_nodes=data.num_nodes,
            num_neg_samples=data.df_mask.sum())

        for epoch in trange(args['epochs'], desc='Unlerning'):
            model.train()

            start_time = time.time()
            z1, z2 = model(data.x, data.edge_index[:, data.sdf_mask], return_all_emb=True)
            # print('current deletion weight', model.deletion1.deletion_weight.sum(), model.deletion2.deletion_weight.sum())
            # print('aaaaaa', z[data.sdf_node_2hop_mask].sum())

            # Randomness
            pos_edge = data.edge_index[:, data.df_mask]
            # neg_edge = torch.randperm(data.num_nodes)[:pos_edge.view(-1).shape[0]].view(2, -1)

            embed1 = torch.cat([z1[pos_edge[0]], z1[pos_edge[1]]], dim=0)
            embed1_ori = torch.cat([z1_ori[neg_edge[0]], z1_ori[neg_edge[1]]], dim=0)

            embed2 = torch.cat([z2[pos_edge[0]], z2[pos_edge[1]]], dim=0)
            embed2_ori = torch.cat([z2_ori[neg_edge[0]], z2_ori[neg_edge[1]]], dim=0)

            loss_r1 = loss_fct(embed1, embed1_ori)
            loss_r2 = loss_fct(embed2, embed2_ori)

            # Local causality
            loss_l1 = loss_fct(z1[data.sdf_node_1hop_mask_non_df_mask], z1_ori[data.sdf_node_1hop_mask_non_df_mask])
            loss_l2 = loss_fct(z2[data.sdf_node_2hop_mask_non_df_mask], z2_ori[data.sdf_node_2hop_mask_non_df_mask])


            # Total loss
            '''both_all, both_layerwise, only2_layerwise, only2_all, only1'''
            loss_l = loss_l1 + loss_l2
            loss_r = loss_r1 + loss_r2

            loss1 = self.args['alpha'] * loss_r1 + (1 - self.args['alpha']) * loss_l1
            loss1.backward(retain_graph=True)
            if(type(optimizer) is list):
                optimizer[0].step()
                optimizer[0].zero_grad()
            else:
                optimizer.step()
                optimizer.zero_grad()

            loss2 = self.args['alpha'] * loss_r2 + (1 - self.args['alpha']) * loss_l2
            loss2.backward(retain_graph=True)
            if(type(optimizer) is list):
                optimizer[1].step()
                optimizer[1].zero_grad()

            loss = loss1 + loss2

            end_time = time.time()
            epoch_time = end_time - start_time

            step_log = {
                'Epoch': epoch,
                'train_loss': loss.item(),
                'loss_r': loss_r.item(),
                'loss_l': loss_l.item(),
                'train_time': epoch_time
            }
            wandb.log(step_log)
            msg = [f'{i}: {j:>4d}' if isinstance(j, int) else f'{i}: {j:.4f}' for i, j in step_log.items()]
            tqdm.write(' | '.join(msg))

            if (epoch + 1) % self.args['valid_freq'] == 0:
                valid_loss, dt_acc, dt_f1, valid_log = self.eval(model, data, 'val')
                valid_log['epoch'] = epoch

                train_log = {
                    'epoch': epoch,
                    'train_loss': loss.item(),
                    'loss_r': loss_r.item(),
                    'loss_l': loss_l.item(),
                    'train_time': epoch_time,
                }

                for log in [train_log, valid_log]:
                    wandb.log(log)
                    msg = [f'{i}: {j:>4d}' if isinstance(j, int) else f'{i}: {j:.4f}' for i, j in log.items()]
                    tqdm.write(' | '.join(msg))
                    self.trainer_log['log'].append(log)

                if dt_acc + dt_f1 > best_metric:
                    best_metric = dt_acc + dt_f1
                    best_epoch = epoch

                    print(f'Save best checkpoint at epoch {epoch:04d}. Valid loss = {valid_loss:.4f}')
                    ckpt = {
                        'model_state': model.state_dict(),
                        # 'optimizer_state': [optimizer[0].state_dict(), optimizer[1].state_dict()],
                    }
                    torch.save(ckpt, os.path.join(args['checkpoint_dir'], 'model_best.pt'))

        # Save
        ckpt = {
            'model_state': {k: v.to('cpu') for k, v in model.state_dict().items()},
            # 'optimizer_state': [optimizer[0].state_dict(), optimizer[1].state_dict()],
        }
        torch.save(ckpt, os.path.join(args['checkpoint_dir'], 'model_final.pt'))

class DeletionLayer(nn.Module):
    def __init__(self, dim, mask):
        super().__init__()
        self.dim = dim
        self.mask = mask
        self.deletion_weight = nn.Parameter(torch.ones(dim, dim) / 1000)
        # self.deletion_weight = nn.Parameter(torch.eye(dim, dim))
        # init.xavier_uniform_(self.deletion_weight)

    def forward(self, x, mask=None):
        '''Only apply deletion operator to the local nodes identified by mask'''

        if mask is None:
            mask = self.mask

        if mask is not None:
            new_rep = x.clone()
            new_rep[mask] = torch.matmul(new_rep[mask], self.deletion_weight)

            return new_rep

        return x

class GCNDelete(GCN):
    def __init__(self, args, mask_1hop=None, mask_2hop=None, **kwargs):
        super().__init__(args)
        self.deletion1 = DeletionLayer(args['hidden_dim, mask_1hop'])
        self.deletion2 = DeletionLayer(args['out_dim, mask_2hop'])

        self.conv1.requires_grad = False
        self.conv2.requires_grad = False

    def forward(self, x, edge_index, mask_1hop=None, mask_2hop=None, return_all_emb=False):
        # with torch.no_grad():
        x1 = self.conv1(x, edge_index)

        x1 = self.deletion1(x1, mask_1hop)

        x = F.relu(x1)

        x2 = self.conv2(x, edge_index)
        x2 = self.deletion2(x2, mask_2hop)

        if return_all_emb:
            return x1, x2

        return x2

    def get_original_embeddings(self, x, edge_index, return_all_emb=False):
        return super().forward(x, edge_index, return_all_emb)

class GINDelete(GIN):
    def __init__(self, args, mask_1hop=None, mask_2hop=None, **kwargs):
        super().__init__(args)
        self.deletion1 = DeletionLayer(args['hidden_dim, mask_1hop'])
        self.deletion2 = DeletionLayer(args['out_dim, mask_2hop'])

        self.conv1.requires_grad = False
        self.conv2.requires_grad = False

    def forward(self, x, edge_index, mask_1hop=None, mask_2hop=None, return_all_emb=False):
        with torch.no_grad():
            x1 = self.conv1(x, edge_index)

        x1 = self.deletion1(x1, mask_1hop)

        x = F.relu(x1)

        x2 = self.conv2(x, edge_index)
        x2 = self.deletion2(x2, mask_2hop)

        if return_all_emb:
            return x1, x2

        return x2

    def get_original_embeddings(self, x, edge_index, return_all_emb=False):
        return super().forward(x, edge_index, return_all_emb)

class GNNDeletion:
    def __init__(self, args, model, poisoned_dataset):
        self.args = args
        self.dataset = poisoned_dataset
        self.model = model
        self.args['checkpoint_dir'] = 'checkpoint_node'
        self.original_path = os.path.join(self.args['checkpoint_dir'], 'cora', 'gcn', 'original', str(self.args['random_seed']))
        self.attack_path_all = os.path.join(self.args['checkpoint_dir'], 'cora', 'member_infer_all', str(self.args['random_seed']))
        self.attack_path_sub = os.path.join(self.args['checkpoint_dir'], 'cora', 'member_infer_sub', str(self.args['random_seed']))
        seed_everything(self.args['random_seed'])

        self.args['checkpoint_dir'] = os.path.join(
            self.args['checkpoint_dir'], 'cora', 'gcn', f'{"gnndelete"}-node_deletion',
            '-'.join([str(i) for i in [self.args['loss_fct'], self.args['loss_type'], self.args['alpha'], self.args['neg_sample_random']]]),
            '-'.join([str(i) for i in [self.args['df'], self.args['df_size'], self.args['random_seed']]]))

        os.makedirs(self.args['checkpoint_dir'], exist_ok=True)

    def load_dataset(self):
        # dataset = CitationFull(os.path.join(self.args['data_dir, 'cora'), 'cora', transform=T.NormalizeFeatures())
        dataset = self.dataset
        data = dataset[0]
        print('Original data', data)

        split = T.RandomNodeSplit()
        data = split(data)
        assert is_undirected(data.edge_index)

        print('Split data', data)
        self.args['in_dim'] = data.x.shape[1]
        self.args['out_dim'] = dataset.num_classes
        self.data = data
        wandb.init(config=self.args)

    def delete_nodes(self):
        if self.args['df_size'] >= 100:  # df_size is number of nodes/edges to be deleted
            df_size = int(self.args['df_size'])
        else:  # df_size is the ratio
            df_size = int(self.args['df_size'] / 100 * self.data.edge_index.shape[1])

        print(f'Original size: {self.data.num_nodes:,}')
        print(f'Df size: {df_size:,}')

        df_nodes = torch.randperm(self.data.num_nodes)[:df_size]
        global_node_mask = torch.ones(self.data.num_nodes, dtype=torch.bool)
        global_node_mask[df_nodes] = False

        dr_mask_node = global_node_mask
        df_mask_node = ~global_node_mask
        assert df_mask_node.sum() == df_size

        res = [torch.eq(self.data.edge_index, aelem).logical_or_(torch.eq(self.data.edge_index, aelem)) for aelem in df_nodes]
        df_mask_edge = torch.any(torch.stack(res, dim=0), dim=0)
        df_mask_edge = df_mask_edge.sum(0).bool()
        dr_mask_edge = ~df_mask_edge

        df_edge = self.data.edge_index[:, df_mask_edge]
        self.data.directed_df_edge_index = to_directed(df_edge)

        print('Deleting the following nodes:', df_nodes)

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

    def load_model(self):
        here = self.model
        self.model = GCNDelete(self.args)

        if os.path.exists(os.path.join(self.original_path, 'pred_proba.pt')):
            self.logits_ori = torch.load(os.path.join(self.original_path, 'pred_proba.pt'))
            if self.logits_ori is not None:
                self.logits_ori = self.logits_ori.to(device)
        else:
            self.logits_ori = None

        model_ckpt = here
        self.model.load_state_dict(model_ckpt['model_state'], strict=False)

        self.model = self.model.to(device)

    def setup_optimizer(self):
        parameters_to_optimize = [
            {'params': [p for n, p in self.model.named_parameters() if 'del' in n], 'weight_decay': 0.0}
        ]
        print('parameters_to_optimize', [n for n, p in self.model.named_parameters() if 'del' in n])

        self.optimizer = torch.optim.Adam(parameters_to_optimize, lr=self.args['lr'])

        wandb.watch(self.model, log_freq=100)

    def train(self):
        self.attack_model_all = None
        self.attack_model_sub = None

        trainer = GNNDeleteNodeClassificationTrainer(self.args)
        trainer.train(self.model, self.data, self.optimizer, self.args, self.logits_ori, self.attack_model_all, self.attack_model_sub)

    def run(self):
        self.load_dataset()
        self.delete_nodes()
        self.load_model()
        self.setup_optimizer()
        self.train()