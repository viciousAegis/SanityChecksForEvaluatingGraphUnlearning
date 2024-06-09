import torch
from tqdm import tqdm
from time import time
from torchmetrics import AUROC
from sklearn.metrics import classification_report


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

'''
Base methods for training and testing; dataset, method & adverserial attack agnostic
'''

def compute_success_rate(loader, model, target_label=1):
    with torch.no_grad():
        running_acc = 0
        count = 0
        for i, data in enumerate(loader):
            data = data.to(device)
            by = data.y.to(device)

            count += by.size(0)

            logits = model(data)
            running_acc += (torch.max(logits, dim=1)[1] == target_label).float().sum(0).cpu().numpy()
        acc = running_acc / count
    print(f'Trigger success rate: {acc:.4f}')

def train(model, optimizer, criterion, train_loader):
    model.train()
    for data in train_loader:  # Iterate in batches over the training dataset.
        out = model(data.to(device))  # Perform a single forward pass.
        loss = criterion(out, data.y)  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.

def test(model, loader, eval=True):
    model.eval()
    auroc = AUROC(task='binary')
    y_true = []
    y_pred = []
    correct = 0
    for data in loader:  # Iterate in batches over the training/test dataset.
        out = model(data.to(device))  
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        correct += int((pred == data.y).sum()) / len(data) # Check against ground-truth labels.
        y_true.append(data.y)
        y_pred.append(pred)
        # get classification report
    roc = None
    if not eval:
        y_true = torch.cat(y_true, dim=0)
        y_pred = torch.cat(y_pred, dim=0)
        roc = auroc(y_pred, y_true)
    acc = correct / len(loader)
    return acc, roc

def evaluate_model(model, train_loader, test_loader, iters=300, lr = 0.001):
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in tqdm(range(1, iters)):
        train(model, optimizer, criterion, train_loader)
        train_acc, _ = test(model, train_loader)
        test_acc, _ = test(model, test_loader)
    print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
    return model

'''
high level train/eval methods
'''

def train_model(model, train_loader, test_loader):
    start_time = time()
    trained_model = evaluate_model(model=model, train_loader=train_loader, test_loader=test_loader)
    print(f'model training time: {time() - start_time:.2f}s')
    # save the model
    return trained_model

def test_model(model, test_loader):
    test_acc, roc = test(model, test_loader, eval=False)
    print(f'Test Acc: {test_acc:.4f}')
    print(f'ROC: {roc}')
