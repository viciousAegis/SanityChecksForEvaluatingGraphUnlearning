import torch
from gub.dataset import load_dataset
from gub.attacks import load_attack
from gub.config import args
from gub.models import load_model
from gub.unlearn import init_unlearn_algo
from gub.train import init_trainer

seed = 1235
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
gen = torch.Generator()
gen.manual_seed(seed)

if __name__ == "__main__":
    dataset = load_dataset(args.dataset_name)

    model = load_model(
        model_name=args.model,
        in_features=dataset.num_features,
        out_features=dataset.num_classes,
        hidden_features=args.hidden_features,
        n_layers=args.n_layers,
    )

    trainer = init_trainer(
        task_level="node",
        dataset=dataset,
        optimizer=torch.optim.Adam(model.parameters(), lr=args.lr_optimizer),
        loss=torch.nn.CrossEntropyLoss(),
        lr_scheduler=True,
        early_stop=True,
    )

    trainer.train(
        model=model,
        n_epoch=args.n_epoch_train,
        verbose=False,
    )

    print(trainer.evaluate(model=model, mask=dataset.test_mask))

    attack = load_attack(
        attack_name=args.attack,
        dataset=dataset,
        device=args.device,
        target_label=args.target_label,
    )

    poisoned_dataset = attack.attack(
        poison_ratio=args.poison_ratio, trigger_size=args.trigger_size
    )

    poisoned_model = load_model(
        model_name=args.model,
        in_features=dataset.num_features,
        out_features=dataset.num_classes,
        hidden_features=args.hidden_features,
        n_layers=args.n_layers,
    )

    poison_trainer = init_trainer(
        task_level="node",
        dataset=dataset,
        optimizer=torch.optim.Adam(poisoned_model.parameters(), lr=args.lr_optimizer),
        loss=torch.nn.CrossEntropyLoss(),
        lr_scheduler=True,
        early_stop=True,
    )

    # poison a model

    poison_trainer.train(
        model=poisoned_model,
        n_epoch=args.n_epoch_train,
        verbose=False,
    )

    # evaluate the poisoned model
    print("Evaluation of the poisoned model on clean test data:")
    print(
        poison_trainer.evaluate(model=poisoned_model, mask=poisoned_dataset.test_mask)
    )

    unlearn_algo = init_unlearn_algo(
        args.unlearn_method, model=poisoned_model, dataset=poisoned_dataset
    )
    unlearn_algo.set_nodes_to_unlearn(poisoned_dataset)
    unlearn_algo.set_unlearn_request(args.unlearn_request)
    
    unlearned_model = unlearn_algo.unlearn()
    
    print("Evaluation of the unlearned model on clean test data:")
    print(trainer.evaluate(model=unlearned_model, mask=dataset.test_mask))
