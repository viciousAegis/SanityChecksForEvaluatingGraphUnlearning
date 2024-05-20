from grb.attack.injection.fgsm import FGSM
from grb.attack.injection.pgd import PGD
from grb.attack.injection.tdgia import TDGIA
from grb.attack.injection.rand import RAND
from grb.attack.injection.speit import SPEIT
from grb.utils.normalize import GCNAdjNorm


def attack(model, dataset, attack_type="tdgia"):

    if(attack_type == 'tdgia'):
        attack = TDGIA(
            lr=0.01,
            n_epoch=10,
            n_inject_max=100,
            n_edge_max=100,
            feat_lim_min=-0.9,
            feat_lim_max=0.9,
            sequential_step=0.2,
        )
    elif(attack_type == 'pgd'):
        attack = PGD(
            epsilon=0.3,
            n_epoch=10,
            n_inject_max=20,
            n_edge_max=20,
            feat_lim_min=-0.9,
            feat_lim_max=0.9,
        )
    elif(attack_type == 'fgsm'):
        attack = FGSM(
            epsilon=0.3,
            n_epoch=10,
            n_inject_max=20,
            n_edge_max=20,
            feat_lim_min=-0.9,
            feat_lim_max=0.9,
        )
    elif(attack_type == 'rand'):
        attack = RAND(
            n_inject_max=20,
            n_edge_max=20,
            feat_lim_min=-0.9,
            feat_lim_max=0.9,
        )
    elif(attack_type == 'speit'):
        attack = SPEIT(
            lr=0.01,
            n_epoch=10,
            n_inject_max=20,
            n_edge_max=20,
            feat_lim_min=-0.9,
            feat_lim_max=0.9,
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
