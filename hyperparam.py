import os

if __name__=="__main__":
    attacks = ["tdgia", "fgsm", "pgd", "rand", "speit"]
    
    # # megu hyperparameters
    # kappa = [0.00001, 0.0001, 0.01, 0.1]
    # alpha1 = [0.1, 0.3, 0.5, 0.7, 0.9]
    # alpha2 = [0.1, 0.3, 0.5, 0.7, 0.9]
    # unlearn_lr = [1e-6,1e-5,1e-4,1e-3]
    
    for attack in attacks:
        print(f"Attack={attack}")
        os.system(f"python main.py --attack {attack} --wandb")
    
    exit()

    # gif hyperparameters
    iteration = [2, 3, 4, 5, 6]
    damp = [0, 0.2, 0.4, 0.6, 0.7]
    scale = [25, 50, 100, 200, 300]

    for attack in attacks:
        for i in iteration:
            for d in damp:
                for s in scale:
                    print(f"Attack={attack} with iteration={i}, damp={d}, scale={s}")

                    os.system(f"python main.py --attack {attack} --wandb --unlearn_method gif --iteration {i} --damp {d} --scale {s}")