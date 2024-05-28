import subprocess

if __name__=="__main__":
    attacks = ["tdgia", "fgsm", "pgd", "rand", "speit"]
    
    # megu hyperparameters
    kappa = [0.00001, 0.0001, 0.01, 0.1]
    alpha1 = [0.1, 0.3, 0.5, 0.7, 0.9]
    alpha2 = [0.1, 0.3, 0.5, 0.7, 0.9]
    unlearn_lr = [1e-6,1e-5,1e-4,1e-3]
    
    
    
    # for attack in attacks:
    #     subprocess.run(["python", "main.py", "--attack", attack, "--wandb", "--unlearn_method", "megu"])
    #     print(f"Attack {attack} completed")
    
    for attack, k, a1, a2, lr in zip(attacks, kappa, alpha1, alpha2, unlearn_lr):
        subprocess.run(["python", "main.py", "--attack", attack, "--wandb", "--unlearn_method", "megu", "--kappa", str(k), "--alpha1", str(a1), "--alpha2", str(a2), "--megu_unlearn_lr", str(lr)])
        print(f"Attack {attack} completed")