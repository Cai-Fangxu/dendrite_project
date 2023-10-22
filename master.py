import os
import sys

n_repeat = 6
seed = 123456
"params_list: b, kappa. ndR"
params_list = [[b, k, ndR] for b in {2.75} for ndR in {2} for k in {2.5}]

for param in params_list:
    for i in range(n_repeat):  
        print("############################################################################", flush=True)  
        os.system(f"python ReLU_parameter_test.py {param[0]} {param[1]} {param[2]} {seed+i}")
