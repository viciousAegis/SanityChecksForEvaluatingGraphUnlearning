

from grb.attack.injection.fgsm import FGSM
from grb.attack.injection.pgd import PGD
from grb.attack.injection.tdgia import TDGIA
from grb.utils.normalize import GCNAdjNorm


def get_attack(name):
    if name == "tdgia":
        return TDGIA
    elif name == "pgd":
        return PGD
    elif name == "fgsm":
        return FGSM


def attack(model, dataset, attack_type="tdgia"):

    attack = get_attack(attack_type)(
        lr=0.01,
        n_epoch=10,
        n_inject_max=20,
        n_edge_max=20,
        feat_lim_min=-0.9,
        feat_lim_max=0.9,
        sequential_step=0.2,
    )
    adj = dataset.adj.tocoo()
    
    poisoned_adj, poisoned_x = attack.attack(
        model=model,
        adj=adj,
        features=dataset.features,
        target_mask=dataset.test_mask,
        adj_norm_func=GCNAdjNorm,
    )
    
    return poisoned_adj, poisoned_x