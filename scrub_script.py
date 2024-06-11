import os

msteps = [10,25,50]
alpha = [0.5,1,2]
lr = [0.025, 0.01, 0.001]

for m in msteps:
    for a in alpha:
        for l in lr:
            os.system(f"python -W ignore scrub_cora.py --msteps {m} --alpha {a} --unlearn_lr {l}")