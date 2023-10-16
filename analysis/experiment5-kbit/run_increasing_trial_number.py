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

num_alph   = 1;
num_ners   = 3;
num_exps   = 20;
num_trials = np.array([25,50,100,200,400,1000,200*10,200*30])
ner_all    = np.array([100,1000,3000]);
alph_all   = np.array([0.5]);
algs = ["CPU","32bit","64bit"]

slopes     = np.zeros([3,num_alph,num_ners,num_exps,num_trials.shape[0]])+ np.nan
cor_p      = np.zeros([3,num_alph,num_ners,num_exps,num_trials.shape[0]])+ np.nan
cor_s      = np.zeros([3,num_alph,num_ners,num_exps,num_trials.shape[0]])+ np.nan
times      = np.zeros([4,num_alph,num_ners,num_exps,num_trials.shape[0]])+ np.nan
out_r2     = np.zeros([3,num_alph,num_ners,num_exps,num_trials.shape[0]])+ np.nan

for i in [0]:
    alph = alph_all[i]
    for j in [2,1,0]:
        n_rec = ner_all[j]
        opts = {'alpha' : alph,
                'n_in' : 3,
                'sigma_input' : 1e-2,
                'sigma_conversion' : 1e-3,
                'verbose' : 0,
                'n_rec' : n_rec}
        for k in range(num_exps):
            start_time = time_now.perf_counter()
            data,rnn_gnd,trial_info =  get_data_from_kbit_task(\
                                        200*30,alph,n_rec,
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
            for t in range(num_trials.shape[0]):
                T_data     = 100 * num_trials[t]
                for l in range(4):
                    r_in       = r_in_all[:T_data,:];
                    r_out      = r_out_all[:T_data,:];
                    u_in       = u_all[:T_data,:];
                    
                    gnd = rnn_gnd['w_rec'].T.flatten()
                    w_out = rnn_gnd['w_out']
                    w_rec_gt = rnn_gnd['w_rec'].T
                    w_in_gt = rnn_gnd['w_in'].T
                    
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
                        times[l,i,j,k,t] = approximate_newton(r_in,r_out,u_in = u_in, alph = alph, 
                                        l2 = 1e-5)
                        temp = time_now.localtime()
                        current_time = time_now.strftime("%H:%M:%S", temp)
                        print('%s: Alpha %.1f. Ner %d. Exp %d. T_data %d. Newton: Time %.2f. .'\
                          %(current_time,alph,n_rec,k,T_data,\
                            times[l,i,j,k,t]))
                        continue
                    try:
                        times[l,i,j,k,t] = time_now.perf_counter() - start_time;
                        prd = w[:opts['n_rec'],:].T.flatten()
                        
                        slopes[l,i,j,k,t] = (gnd @ prd) / (gnd@gnd)
                        
                        cor_p[l,i,j,k,t]= pearsonr(prd,gnd)[0]
                        cor_s[l,i,j,k,t] = spearmanr(prd,gnd)[0]
                    except:
                        continue
                    
                    w_rec_cornn = w[:n_rec,:].T
                    w_in_cornn = w[n_rec:,:].T
                    
                    
                    num_test = 10;
                    temp_test = np.zeros(num_test)
                    for ntest in range(num_test):
                        T_test = 1000
                        
                        opts_new = {};
                        opts_new['n_rec'] = n_rec
                        opts_new['n_in'] = 3
                        opts_new['alpha'] = alph
                        opts_new['sigma_input'] = 1e-2
                        opts_new['sigma_conversion'] =1e-3
                        
                        m1 = RNN(opts_new)
                        m1.rnn['w_rec'] = w_rec_gt
                        m1.rnn['w_in'] = w_in_gt
                        
                        m2 = RNN(opts)
                        m2.rnn['w_rec'] = w_rec_cornn
                        m2.rnn['w_in'] = w_in_cornn
                        
                        inputs, outputs = generate_test_data(T_test)
                        
                        
                        r_gt = m1.get_time_evolution(T = inputs.shape[0], u =inputs)
                        r_cornn = m2.get_time_evolution(T = inputs.shape[0], u =inputs,r_in = r_gt[0,:])
                        
                        z_out_gt    = r_gt[1:,:] @ w_out
                        z_out_cornn = r_cornn[1:,:] @ w_out
                        temp_test[ntest] = pearsonr(z_out_gt.flatten(), \
                                                    z_out_cornn.flatten())[0];
                    out_r2[l,i,j,k,t]= np.mean(temp_test)
                    
                    temp = time_now.localtime()
                    current_time = time_now.strftime("%H:%M:%S", temp)
                    
                    print('%s: Cornn %s Alpha %.1f. Ner %d. Exp %d. T_data %d: Time %.2f. Cor %.2f. Scale %.2f. Outcor %.2f'\
                      %(current_time,algs[l],alph,n_rec,k,T_data,\
                        times[l,i,j,k,t],cor_p[l,i,j,k,t],
                        slopes[l,i,j,k,t],out_r2[l,i,j,k,t]))
                
                
 


np.savez('experiment5_results.npz',slopes = slopes,
         cor_p = cor_p,cor_s = cor_s,
         times = times,out_r2 = out_r2)
