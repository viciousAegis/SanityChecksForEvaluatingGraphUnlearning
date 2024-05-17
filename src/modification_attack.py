
def attack(model, dataset, type="dice"):

    adj = dataset.adj
    test_mask = dataset.test_mask
    labels = dataset.labels
    features=dataset.features

    if(type == "dice"):
        from grb.attack.modification.dice import DICE
        n_edge_test = adj[test_mask].getnnz()
        n_mod_ratio = 0.3
        n_edge_mod = int(n_edge_test * n_mod_ratio)
        ratio_delete = 0.6
        attack = DICE(n_edge_mod, ratio_delete)
        adj_attack = attack.attack(adj, dataset.index_test, labels)

    if(type == "fga"):
        from grb.attack.modification.fga import FGA
        n_edge_test = adj[test_mask].getnnz()
        n_mod_ratio = 0.1
        n_edge_mod = int(n_edge_test * n_mod_ratio)
        attack = FGA(n_edge_mod)
        adj_attack = attack.attack(model=model, adj=adj, features=features, index_target=dataset.index_test)

    return adj_attack