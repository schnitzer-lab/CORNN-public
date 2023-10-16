#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 17:52:09 2022

@author: dinc
"""

import numpy as np
import torch
import torch.nn as nn
from numpy.linalg import inv
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from rnn_class import RNN
from utils_admm import solve_corrn_admm_gpu
from utils_admm import solve_corrn_admm
from utils_kbit import get_data_from_kbit_task
from utils_kbit import generate_test_data
from utils_admm import approximate_newton
import time as time_now
from sklearn.metrics import r2_score

num_alph   = 1;
num_exps   = 10;
alph_all   = np.array([0.5]);
algs = ["CPU","32bit","64bit"]
num_ners  = np.logspace(np.log10(101),np.log10(3000),8).astype(int)

times      = np.zeros([4,num_exps,num_ners.shape[0]])+ np.nan
r2_ner     = np.zeros([3,num_exps,num_ners.shape[0]])+ np.nan


alph = 0.5
opts = {'alpha' : alph,
    'n_in' : 3,
    'sigma_input' : 1e-2,
    'sigma_conversion' : 1e-3,
    'verbose' : 0,
    'n_rec' : 3000}
for k in range(num_exps):
    start_time = time_now.perf_counter()
    data,rnn_gnd,trial_info =  get_data_from_kbit_task(\
                                200,alph,3000,
                                np.mod(k,10),opts)
    tt = time_now.perf_counter() - start_time;
    temp = time_now.localtime()
    current_time = time_now.strftime("%H:%M:%S", temp)
    print('%s: Data collection took %.2f mins.'\
       %(current_time,tt/60))
    
    r_in_all = data['r_in']
    r_out_all = data['r_out']
    u_all = data['u']
    _ = solve_corrn_admm_gpu(r_in_all[:100,:],r_out_all[:100,:],u_all[:100,:], alph = alph, 
                    l2 = 1e-5, threshold = 1, rho = 100,
                    verbose = 0, float_type = '32bit',
                    num_iters = 50,gnd = None,solver_type = 'weighted')
    for t in range(num_ners.shape[0]):
        num_n = num_ners[t]
        for l in range(4):
            r_in       = r_in_all[:,:num_n];
            r_out      = r_out_all[:,:num_n];
            u_in       = u_all.copy();
            
            gnd = rnn_gnd['w_rec'].T
            w_rec_gt = rnn_gnd['w_rec'].T
            w_in_gt = rnn_gnd['w_in'].T
            gnd = gnd[:,:num_n]
            gnd = gnd[:num_n,:].flatten()
            
            start_time = time_now.perf_counter()
            if l == 0:
                try:
                    w = solve_corrn_admm(r_in,r_out,u_in = u_in, alph = alph, 
                                l2 = 1e-5, threshold = 1, rho = 100,
                                verbose = 0,
                                num_iters = 50,gnd = gnd,solver_type = 'weighted')
                except:
                    print('Cornn CPU failed')
                    continue
            elif l == 1:
                try:
                    w = solve_corrn_admm_gpu(r_in,r_out,u_in = u_in, alph = alph, 
                                l2 = 1e-5, threshold = 1, rho = 100,
                                verbose = 0, float_type = '32bit',
                                num_iters = 50,gnd = gnd,solver_type = 'weighted')
                except:
                    print('Cornn GPU 32 bit failed')
                    continue
            elif l == 2:
                try:
                    w = solve_corrn_admm_gpu(r_in,r_out,u_in = u_in, alph = alph, 
                                l2 = 1e-5, threshold = 1, rho = 100,
                                verbose = 0, float_type = '64bit',
                                num_iters = 50,gnd = gnd,solver_type = 'weighted')
                except:
                    print('Cornn GPU 64 bit failed')
                    continue
            else:
                times[l,k,t] = approximate_newton(r_in,r_out,u_in = u_in, alph = alph, 
                                l2 = 1e-5)
                temp = time_now.localtime()
                current_time = time_now.strftime("%H:%M:%S", temp)
                continue
            try:
                times[l,k,t] = time_now.perf_counter() - start_time;
                prd = w[:num_n,:].T.flatten()
            except:
                continue
            
            w_rec_cornn = w[:num_n,:].T
            w_in_cornn = w[num_n:,:].T
            
            num_test = 5;
            temp_test = np.zeros(num_test)
            for ntest in range(num_test):
                T_test = 1000
                

                
                m1 = RNN(opts)
                m1.rnn['w_rec'] = w_rec_gt
                m1.rnn['w_in'] = w_in_gt
                
                opts_new = {};
                opts_new['n_rec'] = num_n
                opts_new['n_in'] = 3
                opts_new['alpha'] = alph
                opts_new['sigma_input'] = 1e-2
                opts_new['sigma_conversion'] =1e-3
                
                m2 = RNN(opts_new)
                m2.rnn['w_rec'] = w_rec_cornn
                m2.rnn['w_in'] = w_in_cornn
                
                inputs, outputs = generate_test_data(T_test)
                
                
                r_gt = m1.get_time_evolution(T = inputs.shape[0], u =inputs)
                r_cornn = m2.get_time_evolution(T = inputs.shape[0], u =inputs,r_in = r_gt[0,:num_n])
                
                r_gt = r_gt[:,:num_n]
                temp_test[ntest] = r2_score(r_gt[:,:100],r_cornn[:,:100],multioutput='uniform_average')
            r2_ner[l,k,t]= np.mean(temp_test)
            
            temp = time_now.localtime()
            current_time = time_now.strftime("%H:%M:%S", temp)
            print('%s: Cornn %s Ner %d. Exp %d. Time %.2f. R2-ner %.2f. '\
              %(current_time,algs[l],num_n,k,\
                times[l,k,t],r2_ner[l,k,t]))
        
        
 


np.savez('experiment5_neural_scaling_results.npz',r2_ner = r2_ner,
         times = times)
