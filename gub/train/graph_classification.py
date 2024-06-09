import torch
from gub.models.GCN import GCN
from torch_geometric.loader import DataLoader
from gub.train.graph_eval import train_model, test_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def graph_classification(model, train_loader, test_loader, poisoned=False):
    
    trained_model = None
    if(poisoned):
        print("Evaluating on poisoned train data")
    else:
        print("Evaluating on clean train data for original model")
    trained_model = train_model(model=model, train_loader=train_loader, test_loader=test_loader).to(device)
    
    if(poisoned):
        print('Testing on clean test data on poisoned model')
    else:
        print('Testing on clean test data on original model')
    test_model(trained_model, test_loader)
    return trained_model