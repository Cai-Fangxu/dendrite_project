import lib
import numpy as np
import jax
import jax.numpy as jnp
import scipy
# from scipy.special import erfcinv
import sys
import os


nd=int(float(sys.argv[1]))
ns=int(float(sys.argv[2]))
rho = 1
bias = float(sys.argv[3])
kappa = float(sys.argv[4])
ndR = int(float(sys.argv[5]))
n_votes = None
vote_th = bias
beta = float(sys.argv[6]) # if beta is not needed, set input beta=-1
neuron_choice = sys.argv[7]
# try: 
#     beta = float(sys.argv[7])
# except:
#     beta = -1


seed = int(sys.argv[-1])
rng = np.random.default_rng(seed)

# ndb = 1.
# ndbk = 0.4
# bias = erfcinv(2*ndb/nd)*np.sqrt(2)
# kappa = erfcinv(2*ndbk/nd)*np.sqrt(2) - bias

decay_steps = 30000
n_tested_patterns = 500
initial_steps = 5000

if neuron_choice == "3_2":
    neuron2 = lib.Neuron3_2(n_synapses=ns, n_dendrites=nd, bias=bias, kappa=kappa, ndR=ndR, n_votes=n_votes, vote_th=vote_th, seed=rng.integers(100000))
elif neuron_choice == "10_2":
    neuron2 = lib.Neuron10_2(n_synapses=ns, n_dendrites=nd, bias=bias, kappa=kappa, ndR=ndR, beta=beta, vote_th=vote_th, seed=rng.integers(100000))
else: 
    neuron2 = lib.Neuron10_2_2(n_synapses=ns, n_dendrites=nd, bias=bias, kappa=kappa, ndR=ndR, beta=beta, vote_th=vote_th, seed=rng.integers(100000))

xs_gen = lib.Xs_Generator1(nd, ns, normalized_len=np.sqrt(ns), seed=rng.integers(100000))
# xs_gen = lib.Xs_Generator3_2(nd, ns, rho, normalized_len=np.sqrt(ns), seed=rng.integers(100000))

simulation_run2 = lib.Simulation_Run(neuron2, xs_gen, decay_steps=decay_steps, initial_steps=initial_steps, n_tested_patterns=n_tested_patterns, refresh_every=500, seed=rng.integers(100000))

simulation_run2.run(progress_bar=False)

upper99_2_list = np.sort(simulation_run2.votes_record, axis=0)[int(n_tested_patterns*0.99), :decay_steps]
lower99_2_list = np.sort(simulation_run2.votes_record, axis=0)[int(n_tested_patterns*0.01), :decay_steps]
upper99_2_steady = np.mean(upper99_2_list[-10000:])
mean_2_init = np.mean(simulation_run2.votes_record, axis=0)[0]
print("Y threshold: ", upper99_2_steady)
print("Y mean initial: ", mean_2_init)

tmp = (lower99_2_list - upper99_2_steady)
tmp = np.bitwise_and(tmp[:-1]>0, tmp[1:]<0)
capacity = np.mean(np.argwhere(tmp))
print("capacity: ", capacity)

tmp_array = np.array([[bias, kappa, ndR, capacity]])
# with open(f"data/paraemter_serach/nd600_ns100/nd600_ns100_neuron_{neuron_choice}.txt", "a") as f:
        # np.savetxt(f, tmp_array, fmt=['%.6f', '%.6f', '%.1f', '%.1f'])

with open(f"tmp_data/nd{nd}_ns{ns}_neuron{neuron_choice}.txt", "a") as f:
        np.savetxt(f, tmp_array, fmt=['%.6f', '%.6f', '%.1f', '%.1f'])

