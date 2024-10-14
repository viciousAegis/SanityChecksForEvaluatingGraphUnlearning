# Sanity Checks for Evaluation Graph Unlearning

Code for the paper, Sanity Checks for Evaluation Graph Unlearning, accepted at [CoLLAs 2024 Workshop Track](https://lifelong-ml.cc/Conferences/2024/acceptedpapersandvideos/conf-2024-71)

_use a virtual env!!!_

pip install -e . # to install the grb package
pip install -r requirements.txt # to install the requirements

## Instructions to run the code:

```
usage: main.py [-h] [--attack {tdgia,fgsm,pgd,rand,speit}] [--model {GCN,GIN,GraphSAGE,MLP}] [--load_model LOAD_MODEL]
               [--lr_optimizer LR_OPTIMIZER] [--n_epoch_train N_EPOCH_TRAIN] [--hidden_features HIDDEN_FEATURES [HIDDEN_FEATURES ...]]
               [--n_layers N_LAYERS] [--unlearn_method {megu,gif}] [--unlearn_request {node,edge,feature}] [--lr_attack LR_ATTACK]
               [--epsilon_attack EPSILON_ATTACK] [--n_epoch_attack N_EPOCH_ATTACK] [--n_inject_max N_INJECT_MAX] [--n_edge_max N_EDGE_MAX]

GNN Unlearning on Injection Attack Testing

options:
  -h, --help            show this help message and exit
  --attack {tdgia,fgsm,pgd,rand,speit}
                        Attack type to test (default: fgsm)
  --model {GCN,GIN,GraphSAGE,MLP}
                        Model to use (default: GCN)
  --load_model LOAD_MODEL
                        Load model from previous training (default: False)
  --lr_optimizer LR_OPTIMIZER
                        Learning rate for optimizer (default: 0.01)
  --n_epoch_train N_EPOCH_TRAIN
                        Number of epochs to train (default: 200)
  --hidden_features HIDDEN_FEATURES [HIDDEN_FEATURES ...]
                        Hidden layer sizes (default: [64, 64])
  --n_layers N_LAYERS   Number of layers in the model (default: 3)
  --unlearn_method {megu,gif}
                        Unlearning method to test (default: megu)
  --unlearn_request {node,edge,feature}
                        Unlearning request type (default: node)
  --lr_attack LR_ATTACK
                        learning rate for attack (default = 0.01)
  --epsilon_attack EPSILON_ATTACK
                        Epsilon for attack (default: 0.3)
  --n_epoch_attack N_EPOCH_ATTACK
                        Number of epochs for attack (default: 10)
  --n_inject_max N_INJECT_MAX
                        Max number of nodes to be injected in attack (default: 100)
  --n_edge_max N_EDGE_MAX
                        Max degree of injected nodes in attack (default: 100)
```
