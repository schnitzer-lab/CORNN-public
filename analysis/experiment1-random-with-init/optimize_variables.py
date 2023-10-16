#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 17:52:09 2022

@author: dinc
"""

import numpy as np
from scipy.stats import pearsonr
from rnn_class import RNN
import matplotlib.pyplot as plt
from numpy.linalg import inv
from utils import solve_corrn_gpu
from utils import solve_corrn
from utils import solve_newton_descent
from utils import solve_gradient_descent
from utils import solve_pytorch_gpu
from utils import solve_pytorch
from utils import fit_FORCE
from utils import fit_FORCE_gpu

opts = {};
opts['g'] = 3;
opts['n_rec'] = 200
opts['n_in'] = 0
opts['sigma_input'] = 1e-2
opts['sigma_conversion'] = 1e-4
T_data = 3000;
m1 = RNN(opts)
r = m1.get_time_evolution(T = T_data)
gnd = m1.rnn['w_rec'].flatten()

#%%
w = solve_corrn(r[:-1,:],r[1:,:],u_in = None,l2 = 1*1e-5,
                verbose = 2,gnd = gnd, threshold = 1,
                num_iters = 30,check_convergence = 0,
                solver_type = 'weighted',alph = 0.1)
w_rec_co = w[:opts['n_rec'],:].T.flatten()

#%%
w3  = solve_pytorch_gpu(r[:-1,:],r[1:,:],u_in = None,alph = 0.1,
                         l2 = 1e-8,verbose = 2,gnd = gnd,
                         initialize_fp = 0,num_iters = 5000,
                         learning_rate = 0.01,solver_type = 'logistic')

w_rec_pt = w3[:opts['n_rec'],:].T.flatten()

#%%
w1  = solve_newton_descent(r[:-1,:],r[1:,:],u_in = None,alph = 0.1,
                         l2 = 1e-5,verbose = 1,threshold =0.2,
                         initialize_fp = 0,num_iters = 10,
                         solver_type = 'weighted')

w_rec_nt = w1[:opts['n_rec'],:].T.flatten()

#%%

w4 = fit_FORCE_gpu(r,None,alph = 0.1,
                lam = 100,g_in = 2,verbose = 2,
                initialize_fp = 1,num_iters = 100,
                gnd = gnd,solver_type = 'firing_rates')

w_rec_fc = w4[:opts['n_rec'],:].T.flatten()

#%%
plt.scatter(gnd,w_rec_pt)
plt.scatter(gnd,w_rec_co)


print('NT:', pearsonr(w_rec_nt,gnd)[0])
print('GD:', pearsonr(w_rec_pt,gnd)[0])
print('CoRNN:', pearsonr(w_rec_co,gnd)[0])



