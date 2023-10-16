#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 14 14:08:15 2023

@author: dinc
"""

#load_ext autoreload
import matplotlib.pyplot as plt
from RNN_lib import CustomRNN, K_bit_flip_flop, coherence_task, train
import numpy as np
import torch
import torch.nn as nn
from numpy.linalg import inv
from scipy.stats import pearsonr
from rnn_class import RNN
from utils_admm import solve_corrn_admm_gpu
from utils_admm import solve_corrn_admm
from utils_kbit import get_data_from_kbit_task
from utils_kbit import generate_test_data
import time as time_now

#%% Load the data

num_trials = 6;
alph = 0.9 
n_rec = 100
num_rnn = 0
verbose = 0


opts = {'alpha' : alph,
        'n_in' : 3,
        'sigma_input' : 1e-2,
        'sigma_conversion' : 1e-3,
        'verbose' : 0,
        'n_rec' : n_rec}

start_time = time_now.perf_counter()

data,rnn_gnd,trial_info =  get_data_from_kbit_task(\
                            num_trials,alph,n_rec,
                            num_rnn,opts)

print('%d secs to import %d trials from %d neuron network' \
      %(time_now.perf_counter()-start_time,num_trials,n_rec))
    
r_in = data['r_in']
r_out = data['r_out']
u = data['u']


n_in  = u.shape[1]
T_data = r_in.shape[0]
w_rec = rnn_gnd['w_rec'].T
w_in = rnn_gnd['w_in'].T
w_out = rnn_gnd['w_out']

gnd = w_rec.flatten()

#%%



w = solve_corrn_admm(r_in,r_out,u_in = u, alph = alph, 
                l2 = 1e-5, threshold = 1, rho = 100,
                verbose = 2,
                num_iters = 10,gnd = gnd,solver_type = 'weighted')


w_rec_cornn = w[:n_rec,:].T
w_in_cornn = w[n_rec:,:].T

w_rec_cornn[np.eye(n_rec,dtype = bool)] = 0

prd = w_rec_cornn.flatten()
plt.scatter(gnd,prd)
plt.show()
print(pearsonr(gnd,prd))



#%%



opts = {};
opts['n_rec'] = n_rec
opts['n_in'] = 3
opts['alpha'] = alph
opts['verbose'] = False;
opts['sigma_input'] = 0
opts['sigma_conversion'] =0 

m1 = RNN(opts)
m1.rnn['w_rec'] = w_rec
m1.rnn['w_in'] = w_in

m2 = RNN(opts)
m2.rnn['w_rec'] = w_rec_cornn
m2.rnn['w_in'] = w_in_cornn

T_test = 200

inputs, outputs = generate_test_data(T_test)


r = m1.get_time_evolution(T = inputs.shape[0], u =inputs)
r_cornn = m2.get_time_evolution(T = inputs.shape[0], u =inputs,r_in = r[0,:])


plt.subplot(2,1,1)
pick = 0
plt.plot(r[:,pick],linewidth = 10) 
plt.plot(r_cornn[:,pick],linewidth = 1) 

plt.subplot(2,1,2)
pick = 1
z_out = r[1:,:] @ w_out

z_out_cornn = r_cornn[1:,:] @ w_out

plt.plot(outputs[:,pick])
plt.plot(z_out[:,pick],linewidth = 10)
plt.plot(z_out_cornn[:,pick],linewidth = 1)





