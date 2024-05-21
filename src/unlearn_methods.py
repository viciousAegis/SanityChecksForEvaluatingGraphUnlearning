from unlearn.MEGU import MEGU
from unlearn.Projector import Projector
from unlearn.GIF import ExpGraphInfluenceFunction

def get_unlearn_method(name, **kwargs):
    if name == "projector":
        return Projector(**kwargs)
    elif name=="megu":
        return MEGU(**kwargs)
    elif name=="gif":
        return ExpGraphInfluenceFunction(**kwargs)
    else:
        raise ValueError("Unlearn method not found")