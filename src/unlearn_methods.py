import copy
import time
import numpy as np
import torch
from unlearn.MEGU import MEGU
from unlearn.Projector import Projector

def get_unlearn_method(name, **kwargs):
    if name == "projector":
        return Projector(**kwargs)
    elif name=="megu":
        return MEGU(**kwargs)
    else:
        raise ValueError("Unlearn method not found")
