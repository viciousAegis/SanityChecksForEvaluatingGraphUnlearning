from src.config import args
from src.dataset import load_dataset
from src.model_utils import train_model
from src.unlearn_methods import get_unlearn_method, UnlearnMethod


if __name__ == "__main__":
    dataset = load_dataset()
    model = train_model(dataset, model_name=args.model)

    if args.attack_type == "injection":
        from src.injection_attack import attack
        poisoned_adj, poisoned_x = attack(
            model, dataset, type=args.attack
        )  # attack returns the poisoned adj and features
    elif args.attack_type == "modification":
        from src.modification_attack import attack
        poisoned_adj = attack(
            model, dataset, type=args.attack
        )  # attack returns the poisoned adj

    ## add unlearning here
    method: UnlearnMethod = get_unlearn_method(
        name="projector",
    )  # kwargs to be defined, see function definition
    method.unlearn()
