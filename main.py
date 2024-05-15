from config import args
from dataset import load_dataset
from poison_methods import attack
from model_utils import train_model
from unlearn_methods import get_unlearn_method, UnlearnMethod


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
