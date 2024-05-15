# AdversarialUnlearning

_use a virtual env!!!_

pip install -e . # to install the grb package
pip install -r requirements.txt # to install the requirements

## Instructions to run the code:

```
usage: main.py [-h] [--attack {tdgia,fgsm,pgd}] [--model {GCN,GIN,GraphSAGE,MLP}] [--lr LR] [--n_epoch N_EPOCH] [--hidden_features HIDDEN_FEATURES [HIDDEN_FEATURES ...]]
               [--n_layers N_LAYERS]

GNN Attack Testing

options:
  -h, --help            show this help message and exit
  --attack {tdgia,fgsm,pgd}
                        Attack type to test (default: tdgia)
  --model {GCN,GIN,GraphSAGE,MLP}
                        Model to use (default: GCN)
  --lr LR               Learning rate for optimizer (default: 0.01)
  --n_epoch N_EPOCH     Number of epochs to train (default: 200)
  --hidden_features HIDDEN_FEATURES [HIDDEN_FEATURES ...]
                        Hidden layer sizes (default: [64, 64])
  --n_layers N_LAYERS   Number of layers in the model (default: 3)
```
