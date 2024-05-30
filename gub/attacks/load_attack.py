from .feature import FeatureTriggerAttack

def load_attack(attack_name, dataset, device, target_label):
    if attack_name == "feature_trigger":
        return FeatureTriggerAttack(
            dataset=dataset,
            device=device,
            target_label=target_label
        )
    else:
        raise ValueError("Invalid attack name")