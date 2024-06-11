import os 

alpha1 = [0.1, 0.5, 0.8, 0.24]
alpha2 = [0.1, 0.5, 0.8, 0.12]
kappa = [0.01, 0.1, 0.05]
megu_unlearn_lr = [0.05, 0.09, 0.001]

for a1 in alpha1:
    for a2 in alpha2:
        for k in kappa:
            for mu in megu_unlearn_lr:
                os.system(f"python -W ignore megu_cora.py --alpha1 {a1} --alpha2 {a2} --kappa {k} --megu_unlearn_lr {mu}")