import torch.nn.functional as F
from torchmetrics import Accuracy
from torchmetrics.classification import MulticlassAccuracy, F1Score
import torch


def train_one_epoch(model, data, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    output = model(data.x, data.edge_index)
    loss = criterion(output[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


def test(model, data):
    model.eval()
    logits = model(data.x, data.edge_index)
    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        pred = logits[mask].max(1)[1]
        acc = 0
        if(mask.sum().item() == 0):
            acc = 0
        else:
            acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs

def test_new(model, data):
    model.eval()
    logits = model(data.x, data.edge_index)
    pred = logits[data.test_mask].max(1)[1]
    acc = 0
    if(data.test_mask.sum().item() == 0):
        acc = 0
    else:
        acc = pred.eq(data.y[data.test_mask]).sum().item() / data.test_mask.sum().item()
    
    f1 = F1Score(num_classes=data.y.max().item() + 1, average='micro', task='multiclass')
    f1_score = f1(pred, data.y[data.test_mask])
    return acc, f1_score.item()


def train(model, data, optimizer, criterion, num_epochs):
    for epoch in range(num_epochs):
        loss = train_one_epoch(model, data, optimizer, criterion)
        if epoch % 10 == 0:
            train_acc, val_acc, test_acc = test(model, data)
            print(
                f"Epoch {epoch}: Loss: {loss:.4f}, Train Acc: {train_acc:.2f}, Val Acc: {val_acc:.2f}, Test Acc: {test_acc:.2f}",
                end="\r",
            )
    print()


def evaluate(model, dataset):
    model.eval()
    device = dataset.x.device
    num_classes = len(torch.unique(dataset.y))
    accuracy = MulticlassAccuracy(num_classes=num_classes, average='micro').to(device)

    with torch.no_grad():
        out = model(dataset.x, dataset.edge_index)
        pred = out.argmax(dim=1)
        correct = pred[dataset.test_mask]
        # print(correct)
        target = dataset.y[dataset.test_mask]
        acc = accuracy(correct, target)

    return acc.item()
