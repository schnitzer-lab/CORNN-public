#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 17:52:09 2022

@author: dinc
"""
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["OMP_NUM_THREADS"] = "5"
os.environ['OPENBLAS_NUM_THREADS'] = '5'
os.environ['MKL_NUM_THREADS'] = '5'
import numpy as np
from rnn_class import RNN
from utils import solve_corrn
from utils import solve_corrn_gpu
from utils import solve_pytorch
from utils import solve_pytorch_gpu
from utils import fit_FORCE
import time as time_now


opts = {};
opts['g'] = 3; 
opts['n_rec'] = 5000 # 5000 
opts['n_in'] = 0
opts['sigma_input'] = 1e-2
opts['sigma_conversion'] = 1e-3
opts['alpha'] = 0.1
opts['input_noise_type'] = 'Gaussian'
opts['conversion_noise_type'] = 'Poisson'
opts['verbose'] = False;
opts['lambda_reg'] = 1e-5
opts['num_cores'] = 4
opts['parallel'] = 1
T_data = 30000; # 30000
t_lim = 48;



num_exps = 10;



for idx_exp in range(7):
    temp = time_now.localtime()
    current_time = time_now.strftime("%H:%M:%S", temp)
    print('%s: Running Experiment %d' %(current_time,idx_exp))
    m1 = RNN(opts)
    r = m1.get_time_evolution(T = T_data)
    gnd = m1.rnn['w_rec'].flatten()
    #% Logistic CPU
    _  = solve_pytorch(r[:-1,:],r[1:,:],u_in = None,alph = 0.1,  time_limit =t_lim,
                                         l2 = 1e-8,verbose = 0,gnd = gnd,
                                         initialize_fp = 0,num_iters = 40000,
                                         learning_rate = .01,solver_type = 'logistic',exp_id = idx_exp)
        


    #% Get Pytorch imported
    _ = solve_corrn_gpu(r[:-1,:],r[1:,:],u_in = None,l2 = 1*1e-5,
                verbose = 0,gnd = gnd, threshold = 1,
                num_iters = 0, float_type = '64bit',
                solver_type = 'weighted',exp_id = 30)

    #% CoRNN GPU
                  
    _ = solve_corrn_gpu(r[:-1,:],r[1:,:],u_in = None,l2 = 1*1e-5,
                    verbose = 0,gnd = gnd, threshold = 1, initialize_fp = 1,
                    num_iters = 100, float_type = '64bit',
                    solver_type = 'weighted',exp_id = idx_exp)

    
    #% PYTORCH GPU logistic
    _  = solve_pytorch_gpu(r[:-1,:],r[1:,:],u_in = None,alph = 0.1, time_limit = t_lim,
                                         l2 = 1e-8,verbose = 0,gnd = gnd, 
                                         initialize_fp =0,num_iters = 40000,
                                         learning_rate = .01,solver_type = 'logistic',exp_id = idx_exp)

    #% PYTORCH GPU logistic - init
    _  = solve_pytorch_gpu(r[:-1,:],r[1:,:],u_in = None,alph = 0.1,  time_limit = t_lim,
                                         l2 = 1e-8,verbose = 0,gnd = gnd, 
                                         initialize_fp =1,num_iters = 40000,
                                         learning_rate = .01,solver_type = 'logistic',exp_id = idx_exp)

    #% PYTORCH GPU L2

    _  = solve_pytorch_gpu(r[:-1,:],r[1:,:],u_in = None,alph = 0.1,  time_limit = t_lim,
                                         l2 = 1e-8,verbose = 0,gnd = gnd, 
                                         initialize_fp =0,num_iters = 40000,
                                         learning_rate = .01,solver_type = 'l2',exp_id = idx_exp)


    #% CORNN CPU    

    _ = solve_corrn(r[:-1,:],r[1:,:],u_in = None,l2 = 1*1e-5,  
                                verbose = 0,gnd = gnd, threshold = 1, initialize_fp = 1, 
                                num_iters = 100,check_convergence = 0,
                                solver_type = 'weighted',exp_id = idx_exp)
    
   
    
    #% FORCE firing rates
    _ = fit_FORCE(r,None,alph = 0.1,
                                lam = 3000,g_in = 2,verbose = 0,  time_limit = t_lim,
                                initialize_fp = 0,num_iters = 50,
                                gnd = gnd,solver_type = 'firing_rates',exp_id = idx_exp)
      


       