from src.config import args
from src.dataset import load_dataset
from src.poison_methods import attack
from src.model_utils import train_model
from src.unlearn_methods import get_unlearn_method, UnlearnMethod


if __name__ == "__main__":
    dataset = load_dataset(custom_dataset=args.custom_dataset)
    model = train_model(dataset, model_name=args.model)
    poisoned_adj, poisoned_x = attack(
        model, dataset, attack_type=args.attack
    )  # attack returns the poisoned adj and features

    ## add unlearning here
    method: UnlearnMethod = get_unlearn_method(
        name="projector",
    )  # kwargs to be defined, see function definition
    method.unlearn()
