from gub.dataset import load_dataset
from gub.attacks import load_attack
from gub.config import args
from gub.models import load_model
from gub.unlearn import init_unlearn_algo
from gub.train import init_trainer

if __name__ == "__main__":
    dataset = load_dataset(args.dataset_name)

    model = load_model(
        model_name=args.model,
        in_features=dataset.in_features,
        out_features=dataset.out_features,
        hidden_features=args.hidden_features,
        n_layers=args.n_layers,
    )
    
    trainer = init_trainer(
        model=model,
        dataset=dataset,
        lr=args.lr_optimizer,
        n_epoch=args.n_epoch_train,
        save_dir=args.save_dir,
        save_name=args.save_name,
    )

    attack = load_attack(
        attack_name=args.attack_name,
        dataset=dataset,
        device=args.device,
        target_label=args.target_label,
    )

    poisoned_dataset = attack.attack(poison_ratio=args.poison_ratio, dataset=dataset)

    unlearn_algo = init_unlearn_algo(args.unlearn_algo)
    trainer.train()
