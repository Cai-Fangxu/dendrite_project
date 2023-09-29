import lib
import numpy as np
import jax
import jax.numpy as jnp
import sys
import os

ns=200
nd=300
bias = 0.17
ndR = 13
k = 0.05
la = 0.526754
fA = 0.1
beta1 = 0.95
beta2 = 0.1
beta3 = float(sys.argv[1])
beta4 = float(sys.argv[2])

decay_steps = 10000
n_tested_patterns = 500
initial_steps = 20000

seed = 238
rng = np.random.default_rng(seed)

neuron = lib.Neuron9(n_synapses=ns, n_dendrites=nd, bias=bias, kappa=k, ndR=ndR, fA=fA, w_len=1., beta1=beta1, beta2=beta2, beta3=beta3, beta4=beta4, seed=rng.integers(100000))

xs_gen = lib.Xs_Generator2(fA=fA, ns=ns, normalizedQ=True, seed=rng.integers(100000))

simulation_run = lib.Simulation_Run(neuron, xs_gen, decay_steps=decay_steps, initial_steps=initial_steps, n_tested_patterns=n_tested_patterns, seed=rng.integers(100000))

simulation_run.run()

votes_mean = np.mean(simulation_run.votes_record, axis=0)[:decay_steps]
votes_lower99 = np.sort(simulation_run.votes_record, axis=0)[int(n_tested_patterns*0.01), :decay_steps]
votes_upper99 = np.sort(simulation_run.votes_record, axis=0)[int(n_tested_patterns*0.99), :decay_steps]
n0 = np.mean(votes_upper99[-4000:])
print("n0 =", n0)
n0 = np.ceil(n0)
false_negative_prob = np.mean(simulation_run.votes_record <= n0, axis=0)[:decay_steps]

cross_pos = np.array(np.nonzero(np.bitwise_and((false_negative_prob[1:] >= 0.01), false_negative_prob[:-1] < 0.01))).reshape((-1, ))
capacity = (np.min(cross_pos) + np.max(cross_pos))/2
print("capacity =", capacity)

try:
    os.mkdir(f"data/6/6_{beta3}_{beta4}")
except:
    pass
np.save(f"data/6/6_{beta3}_{beta4}/votes_mean", votes_mean)
np.save(f"data/6/6_{beta3}_{beta4}/votes_lower99", votes_lower99)
np.save(f"data/6/6_{beta3}_{beta4}/votes_upper99", votes_upper99)
np.save(f"data/6/6_{beta3}_{beta4}/false_negative", false_negative_prob)
tmp = np.load("data/6/b3_b4_capacity.npy")
tmp = np.concatenate((tmp, np.array([[beta3, beta4, capacity]]))).reshape((-1, 3))
np.save("data/6/b3_b4_capacity.npy", tmp)