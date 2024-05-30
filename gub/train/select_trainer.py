from .node_trainer import NodeTrainer

def init_trainer(name, **kwargs):
    if name == "node":
        return NodeTrainer(**kwargs)
    else:
        raise ValueError("Trainer not found")