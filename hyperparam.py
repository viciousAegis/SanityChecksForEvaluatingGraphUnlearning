import subprocess

if __name__=="__main__":
    attacks = ["tdgia", "fgsm", "pgd", "rand", "speit"]
    
    for attack in attacks:
        subprocess.run(["python", "main.py", "--attack", attack, "--wandb", "--unlearn_method", "megu"])
        print(f"Attack {attack} completed")