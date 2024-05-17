from grb.utils.normalize import GCNAdjNorm

def attack(model, dataset, type="tdgia"):

    if(type == 'tdgia'):
        from grb.attack.injection.tdgia import TDGIA
        attack = TDGIA(
            lr=0.01,
            n_epoch=10,
            n_inject_max=20,
            n_edge_max=20,
            feat_lim_min=-0.9,
            feat_lim_max=0.9,
            sequential_step=0.2,
        )
    elif(type == 'pgd'):
        from grb.attack.injection.pgd import PGD
        attack = PGD(
            epsilon=0.01,
            n_epoch=10,
            n_inject_max=20,
            n_edge_max=20,
            feat_lim_min=-0.9,
            feat_lim_max=0.9,
        )
    elif(type == 'fgsm'):
        from grb.attack.injection.fgsm import FGSM
        attack = FGSM(
            epsilon=0.01,
            n_epoch=10,
            n_inject_max=20,
            n_edge_max=20,
            feat_lim_min=-0.9,
            feat_lim_max=0.9,
        )
    elif(type == 'rand'):
        from grb.attack.injection.rand import RAND
        attack = RAND(
            n_inject_max=20,
            n_edge_max=20,
            feat_lim_min=-0.9,
            feat_lim_max=0.9,
        )
    elif(type == 'speit'):
        from grb.attack.injection.speit import SPEIT
        attack = SPEIT(
            lr=0.01,
            n_epoch=10,
            n_inject_max=20,
            n_edge_max=20,
            feat_lim_min=-0.9,
            feat_lim_max=0.9,
        )
    else:
        raise NotImplementedError(f"Attack {type} not supported.")

    adj = dataset.adj.tocoo()

    poisoned_adj, poisoned_x = attack.attack(
        model=model,
        adj=adj,
        features=dataset.features,
        target_mask=dataset.test_mask,
        adj_norm_func=GCNAdjNorm,
    )

    return poisoned_adj, poisoned_x
