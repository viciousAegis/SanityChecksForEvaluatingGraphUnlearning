import torch
from src.config import args
from src.dataset import load_dataset
from src.poison_methods import attack
from src.model_utils import train_model, build_model, test_model
from src.unlearn_methods import get_unlearn_method
from src.utils import build_grb_dataset, make_geometric_data
import wandb
import os
from time import time

seed = 1235
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
gen = torch.Generator()
gen.manual_seed(seed)

if __name__ == "__main__":
    if args.wandb:
        
        wandb.login(key="89ab31781eb2ba697ea9aed8d264c4f778a004ef")
        
        wandb.init(project="corr_graph_unlearn")

        config = wandb.config
        config.update(args)

    print("===========Dataset==========")
    dataset = load_dataset(dataset_name=args.dataset_name, mode=args.test_split)
    print(dataset)

    print("===========Base Model Loading==========")

    try:
        model = torch.load(f"./models/base_model/final_{args.model}_base.pt")
    except:
        print("no loaded base model found, training new model")
        model = train_model(
            dataset=dataset,
            lr=args.lr_optimizer,
            n_epoch=args.n_epoch_train,
            model=build_model(
                model_name=args.model,
                in_features=dataset.num_features,
                out_features=dataset.num_classes,
                hidden_features=[64, 64],
                n_layers=3,
            ),
            save_dir=f"./models/base_model",
            save_name=f"{args.model}_base",
        )

    acc = test_model(model, dataset)

    if args.wandb:
        wandb.log({"Base Model Test Accuracy": acc})


    try:
        poisoned_adj = torch.load(f"./attack_result_data/{args.attack}_poisoned_adj.pt")
        poisoned_x = torch.load(f"./attack_result_data/{args.attack}_poisoned_x.pt")
    except:
        print("===========Injection Attack==========")
        poisoned_adj, poisoned_x = attack(
            model, dataset, attack_type=args.attack
        )  # attack returns the poisoned adj and features

        torch.save(poisoned_adj, f"./attack_result_data/{args.attack}_poisoned_adj.pt")
        torch.save(poisoned_x, f"./attack_result_data/{args.attack}_poisoned_x.pt")
        
    poisoned_dataset = build_grb_dataset(poisoned_adj, poisoned_x, dataset)


    print("===========Poisoned Model Loading==========")
    try:
        poison_trained_model = torch.load(f"./models/{args.poison_model_name}/final_{args.poison_model_name}_poisoned.pt")
    except:
        print(f"no loaded model found for {args.attack}, training new model")
        #Model is trained on the poisoned data
        poison_trained_model = train_model(
            dataset=poisoned_dataset,
            lr=args.lr_optimizer,
            n_epoch=args.n_epoch_train,
            model=build_model(
                model_name=args.model,
                in_features=dataset.num_features,
                out_features=dataset.num_classes,
                hidden_features=[64, 64],
                n_layers=3,
            ),
            save_dir=f"./models/{args.poison_model_name}",
            save_name=f"{args.poison_model_name}_poisoned",
        )

    acc = test_model(poison_trained_model, poisoned_dataset)

    if args.wandb:
        wandb.log({"Poisoned Model Test Accuracy": acc})

    print("===========Unlearning==========")
    #Data into geometric
    poisoned_data = make_geometric_data(poisoned_adj, poisoned_x, dataset)

    """
    INTENDED IMPLEMENTATION

    method = get_unlearn_method(name=args.unlearn_method, kwargs=kwargs) // only runs method.__init__(**kwargs)
    method.set_unlearn_request(args.unlearn_request)
    method.set_nodes_to_unlearn(poisoned_set)
    method.unlearn()
    """

    """
    PLEASE REFACTOR THE BELOW INTO THE ABOVE
    """

    args_gnndelete = {
        'model': poison_trained_model,
        'hidden_features': 128,
        'dataset': poisoned_dataset,
        'epochs': 10,
        'valid_freq': 100,
        'checkpoint_dir': './checkpoint',
        'alpha': 0.5,
        'neg_sample_random': 'non_connected',
        'loss_fct': 'mse_mean',
        'loss_type': 'both_layerwise',
        'in_dim': 128,
        'out_dim': 64,
        'random_seed': 42,
        'batch_size': 8192,
        'num_steps': 32,
        'eval_on_cpu': False,
        'df': 'none',
        'df_size': 0.5,
    }

    method = get_unlearn_method(args.unlearn_method, model=poison_trained_model, data=poisoned_data)
    method.set_unlearn_request(args.unlearn_request)
    method.set_nodes_to_unlearn(poisoned_data)
    unlearned_model = method.unlearn()

    
    if args.wandb:
        save_dir=f"./models/{args.poison_model_name}"
        save_name=f"{args.experiment_name}_unlearned_{time()}.pt"
        method.save_unlearned_model(save_dir, save_name)
        artifact = wandb.Artifact(f"{args.experiment_name}_unlearned", type="model")
        artifact.add_file(f"./models/{args.poison_model_name}/{save_name}")
        wandb.log_artifact(artifact)
        os.remove(f"./models/{args.poison_model_name}/{save_name}")
        

    acc = test_model(unlearned_model, poisoned_dataset)

    if args.wandb:
        wandb.log({"Unlearned Model Test Accuracy": acc})

    print("=====================")
