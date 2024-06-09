import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class GCN(nn.Module):
    def __init__(self, num_features, hidden_dim, num_classes, is_graph_classification=True):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(num_features, hidden_dim)
        if is_graph_classification:
            self.conv2 = GCNConv(hidden_dim, hidden_dim)
        else:
            self.conv2 = GCNConv(hidden_dim, num_classes)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.is_graph_classification = is_graph_classification

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)

        x = self.conv2(x, edge_index)
        
        # apply global pooling to get final node embeddings
        if self.is_graph_classification:
            x = F.relu(x)
            x = global_mean_pool(x, data.batch)
            x = self.fc(x)

        return F.log_softmax(x, dim=1)