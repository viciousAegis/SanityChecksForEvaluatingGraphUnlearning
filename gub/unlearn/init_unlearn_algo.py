from .algo import GIF, MEGU, Projector

def init_unlearn_algo(name, model, dataset):
    if name == "projector":
        return Projector(model=model, data=dataset)
    elif name=="megu":
        return MEGU(model=model, data=dataset)
    elif name=="gif":
        return GIF(model=model, data=dataset)
    else:
        raise ValueError("Unlearn method not found")