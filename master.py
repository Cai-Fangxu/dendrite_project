import os
import sys
import numpy as np

n_repeat = 6
seed = 123456
"params_list: nd, ns, b, kappa, ndR, beta"
# params_list = [[b, k, ndR] for b in {2.75, 3.} for ndR in {1, 2} for k in {3.25}]
# params_list = np.genfromtxt("data/paraemter_serach/nd600_ns100/beta_list_tmp.txt") # each param combination = (b, k, ndR, nd, beta)
params_list = np.genfromtxt("tmp_data/param_list.txt")

print(params_list)

for param in params_list:
    for i in range(n_repeat):  
        # print("############################################################################", flush=True)  
        # os.system(f"python ReLU_parameter_test.py {param[0]} {param[1]} {param[2]} 3_2 {param[4]} {seed+i}")
        print("############################################################################", flush=True)  
        os.system(f"python ReLU_parameter_test.py {param[0]} {param[1]} {param[2]} {param[3]} {param[4]} {param[5]} 10_2 {seed+i}")
        # print("############################################################################", flush=True)  
        # os.system(f"python ReLU_parameter_test.py {param[0]} {param[1]} {param[2]} 10_2_2 {param[4]} {seed+i}")

# for param in params_list:
#     for i in range(n_repeat):  
#         print("############################################################################", flush=True)  
#         os.system(f"python ReLU_parameter_test.py {param[0]} {param[1]} {param[2]} 10_2_2 {param[4]} {seed+i}")
