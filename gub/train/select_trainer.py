from .node_trainer import NodeTrainer

def init_trainer(task_level, **kwargs):
    if task_level == "node":
        return NodeTrainer(**kwargs)
    else:
        raise ValueError("Trainer not found")