import json
import torch
from sklearn.metrics import accuracy_score, f1_score
from pathlib import Path
from grb.evaluator.metric import *


def custom_evaluate(model, X, y, test_mask, is_poisoning):
    """
    For poisoning, we use GRB's data format, which has slightly diff namings of attributes, features instead of x, lables instead of y. 
    So for poisoning, we're using the eval functions written in grb, for unlearning, i'm assuming we're using pyg's deault format, and hence scikit learn metrics
    """
    model.eval() 
    with torch.no_grad():
        logits = model(X[test_mask])
        preds = logits.argmax(dim=1)
    
    true_labels = y[test_mask]

    if is_poisoning:
        # Use custom evaluation metrics if this is for poisoning
        accuracy = eval_acc(preds, true_labels)  
        f1 = eval_f1multilabel(preds, true_labels)  
    else:
        accuracy = accuracy_score(true_labels.cpu().numpy(), preds.cpu().numpy())
        f1 = f1_score(true_labels.cpu().numpy(), preds.cpu().numpy(), average='weighted')

    support_original_class_0 = (true_labels == 0).sum().item()
    support_pred_class_0 = (preds == 0).sum().item()

    return { 
        "Accuracy": accuracy,
        "F1 Score": f1,
        "Support of Class 0 in True Labels": support_original_class_0,
        "Support of Class 0 in Predictions": support_pred_class_0
    }

def report_results(report, save_results=False, save_path=None):
    """
    Prints and optionally saves the evaluation report.
    """
    print("Evaluation Report:")
    for key, value in report.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    
    if save_results:
        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with save_path.open('w') as file:
                json.dump(report, file, indent=4)
        else:
            print("No save path provided. Results will not be saved.")
