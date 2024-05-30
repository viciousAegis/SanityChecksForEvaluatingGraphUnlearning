from .algo import GIF, MEGU, Projector, GNNDelete

def init_unlearn_algo(name, **kwargs):
    if name == "projector":
        return Projector(**kwargs)
    elif name=="megu":
        return MEGU(**kwargs)
    elif name=="gif":
        return GIF(**kwargs)
    elif name=="gnndelete":
        return GNNDelete(**kwargs)
    else:
        raise ValueError("Unlearn method not found")