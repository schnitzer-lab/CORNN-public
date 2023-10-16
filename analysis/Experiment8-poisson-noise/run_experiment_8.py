#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 17:52:09 2022

@author: dinc
"""

import numpy as np
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from rnn_class import RNN
import matplotlib.pyplot as plt
from numpy.linalg import inv
from utils import solve_corrn
from utils import solve_corrn_gpu
from utils import solve_newton_descent
from utils import solve_gradient_descent
from utils import solve_pytorch
from utils import solve_pytorch_gpu
from utils import fit_FORCE
import time as time_now

time_limit = 120;

opts = {};
opts['g'] = 3; 
opts['n_rec'] = 200 # 300 for short
opts['n_in'] = 1
opts['sigma_input'] = 1e-2
opts['sigma_conversion'] = 1e-1
opts['alpha'] = 0.1
opts['input_noise_type'] = 'Poisson'
opts['conversion_noise_type'] = 'Poisson'
opts['verbose'] = False;
opts['lambda_reg'] = 1e-5
opts['num_cores'] = 4
opts['parallel'] = 1
T_data = 1500; # 3000 for short


num_algs = 9;
num_exps = 100;
iter_list = np.linspace(0,20,21).astype(int)
slopes = np.zeros([num_algs,num_exps,iter_list.shape[0]])+ np.nan
rmse = np.zeros([num_algs,num_exps,iter_list.shape[0]])+ np.nan
rmedse = np.zeros([num_algs,num_exps,iter_list.shape[0]])+ np.nan
cor_p = np.zeros([num_algs,num_exps,iter_list.shape[0]])+ np.nan
cor_s = np.zeros([num_algs,num_exps,iter_list.shape[0]])+ np.nan
times = np.zeros([num_algs,num_exps,iter_list.shape[0]])+ np.nan


for idx_exp in range(num_exps):
    counter = np.zeros(num_algs)
    temp = time_now.localtime()
    current_time = time_now.strftime("%H:%M:%S", temp)
    print('%s: Running Experiment %d' %(current_time,idx_exp))
    m1 = RNN(opts)
    r = m1.get_time_evolution(T = T_data)
    gnd = m1.rnn['w_rec'].flatten()
    #%
    for iters in iter_list:
        temp = time_now.localtime()
        current_time = time_now.strftime("%H:%M:%S", temp)
        print('\t %s: Running iteration %d' %(current_time,iters))
        
        if iters == 0:
            try:
                w  = solve_newton_descent(r[:-1,:],r[1:,:],u_in = None,alph = 0.1,
                                     l2 = 1e-5,verbose = 0,threshold =0.5,
                                     initialize_fp = 1,num_iters = iters,
                                     solver_type = 'weighted')
            except:
                w = 0
        
        start_time = time_now.perf_counter()
        if counter[3] == 0:
            try:
                w  = solve_newton_descent(r[:-1,:],r[1:,:],u_in = None,alph = 0.1,
                                         l2 = 1e-5,verbose = 0,threshold =0.5,
                                         initialize_fp = 1,num_iters = iters,
                                         solver_type = 'weighted')
                times[3,idx_exp,iters] = time_now.perf_counter() - start_time;
                w_rec = w[:opts['n_rec'],:].T.flatten()
                slopes[3,idx_exp,iters] = (gnd @ w_rec) / (gnd@gnd)
                rmse[3,idx_exp,iters] = np.sqrt(np.mean( (w_rec-gnd)**2 ))
                rmedse[3,idx_exp,iters] = np.sqrt(np.median( (w_rec-gnd)**2 ))
                cor_p[3,idx_exp,iters] = pearsonr(w_rec,gnd)[0]
                cor_s[3,idx_exp,iters] = spearmanr(w_rec,gnd)[0]
                print('\t \t \t NT,weighted. Cor %.3f' %(cor_p[3,idx_exp,iters]))
                if time_now.perf_counter() - start_time > time_limit:
                    counter[3] =1
            except:
                w = 0
        else:
            print('\t \t \t NT, weighted skipped due to timeout error' )
            

        
        
        start_time = time_now.perf_counter()
        if counter[5] == 0: 
            try:
                w  = solve_newton_descent(r[:-1,:],r[1:,:],u_in = None,alph = 0.1,
                                         l2 = 1e-5,verbose = 0,threshold =0.5,
                                         initialize_fp = 0,num_iters = iters,
                                         solver_type = 'logistic')
                times[5,idx_exp,iters] = time_now.perf_counter() - start_time;
                w_rec = w[:opts['n_rec'],:].T.flatten()
                slopes[5,idx_exp,iters] = (gnd @ w_rec) / (gnd@gnd)
                rmse[5,idx_exp,iters] = np.sqrt(np.mean( (w_rec-gnd)**2 ))
                rmedse[5,idx_exp,iters] = np.sqrt(np.median( (w_rec-gnd)**2 ))
                cor_p[5,idx_exp,iters] = pearsonr(w_rec,gnd)[0]
                cor_s[5,idx_exp,iters] = spearmanr(w_rec,gnd)[0]
                print('\t \t \t NT,logistic. Cor %.3f' %(cor_p[5,idx_exp,iters]))
                if time_now.perf_counter() - start_time > time_limit:
                    counter[5] =1
            except:
                print('\t \t \t NT,logistic terminated due to timeout error')
                counter[5] = 1
        else:
            print('\t \t \t NT,logistic skipped due to timeout error')
            

            
         

        
        
        #%
        start_time = time_now.perf_counter()
        if counter[0] == 0:
            try:
                 
                w = solve_corrn(r[:-1,:],r[1:,:],u_in = None,l2 = 1*1e-5,
                                verbose = 0,gnd = gnd, threshold = 0.5,
                                num_iters = 5*iters,check_convergence = 0,
                                solver_type = 'weighted')
                times[0,idx_exp,iters] = time_now.perf_counter() - start_time;
                w_rec = w[:opts['n_rec'],:].T.flatten()
                slopes[0,idx_exp,iters] = (gnd @ w_rec) / (gnd@gnd) 
                rmse[0,idx_exp,iters] = np.sqrt(np.mean( (w_rec-gnd)**2 ))
                rmedse[0,idx_exp,iters] = np.sqrt(np.median( (w_rec-gnd)**2 ))
                cor_p[0,idx_exp,iters] = pearsonr(w_rec,gnd)[0]
                cor_s[0,idx_exp,iters] = spearmanr(w_rec,gnd)[0]
                print('\t \t \t CoRNN, CPU. Cor %.3f' %(cor_p[0,idx_exp,iters]))
                if time_now.perf_counter() - start_time > time_limit:
                    counter[0] =1
            except:
                print('\t \t \t CoRNN, CPU terminated due to timeout error')
                counter[0] = 1
        else:
            print('\t \t \t CoRNN, CPU skipped due to timeout error')
        
                
        #%
        if iters == 0:
            w = solve_corrn_gpu(r[:-1,:],r[1:,:],u_in = None,l2 = 1*1e-5,
                            verbose = 0,gnd = gnd, threshold = 0.5,
                            num_iters = 5*iters,
                            solver_type = 'weighted')
        start_time = time_now.perf_counter()
        if counter[8] == 0:
            try:
                  
                w = solve_corrn_gpu(r[:-1,:],r[1:,:],u_in = None,l2 = 1*1e-5,
                                verbose = 0,gnd = gnd, threshold = 0.5,
                                num_iters = 5*iters,  float_type = '64bit',
                                solver_type = 'weighted')
                times[8,idx_exp,iters] = time_now.perf_counter() - start_time;
                w_rec = w[:opts['n_rec'],:].T.flatten()
                slopes[8,idx_exp,iters] = (gnd @ w_rec) / (gnd@gnd) 
                rmse[8,idx_exp,iters] = np.sqrt(np.mean( (w_rec-gnd)**2 ))
                rmedse[8,idx_exp,iters] = np.sqrt(np.median( (w_rec-gnd)**2 ))
                cor_p[8,idx_exp,iters] = pearsonr(w_rec,gnd)[0]
                cor_s[8,idx_exp,iters] = spearmanr(w_rec,gnd)[0]
                print('\t \t \t CoRNN, GPU. Cor %.3f' %(cor_p[8,idx_exp,iters]))
                if time_now.perf_counter() - start_time > time_limit:
                    counter[8] =1
            except:
                print('\t \t \t CoRNN, GPU terminated due to timeout error')
                counter[8] = 1
        else:
            print('\t \t \t CoRNN, GPU skipped due to timeout error')
        
        #%
        start_time = time_now.perf_counter()
        if counter[1] == 0:
            try:
                
                w  = solve_pytorch_gpu(r[:-1,:],r[1:,:],u_in = None,alph = 0.1,
                                         l2 = 1e-8,verbose = 0,gnd = gnd, #1e-7 for short
                                         initialize_fp =0,num_iters = iters*1000,
                                         learning_rate = .01,solver_type = 'logistic')
                times[1,idx_exp,iters] = time_now.perf_counter() - start_time;
                w_rec = w[:opts['n_rec'],:].T.flatten()
                slopes[1,idx_exp,iters] = (gnd @ w_rec) / (gnd@gnd)
                rmse[1,idx_exp,iters] = np.sqrt(np.mean( (w_rec-gnd)**2 ))
                rmedse[1,idx_exp,iters] = np.sqrt(np.median( (w_rec-gnd)**2 ))
                cor_p[1,idx_exp,iters] = pearsonr(w_rec,gnd)[0]
                cor_s[1,idx_exp,iters] = spearmanr(w_rec,gnd)[0]
                print('\t \t \t Pytorch,logistic,GPU. Cor %.3f' %(cor_p[1,idx_exp,iters]))
                if time_now.perf_counter() - start_time > time_limit:
                    counter[1] =1
            except:
                print('\t \t \t Pytorch,logistic,GPU terminated due to timeout error')
                counter[1] = 1
        else:
            print('\t \t \t Pytorch,logistic,GPU skipped due to timeout error')
        
        
        #%
        start_time = time_now.perf_counter()
        if counter[2] == 0:
            try:
                 
                w  = solve_pytorch(r[:-1,:],r[1:,:],u_in = None,alph = 0.1,
                                         l2 = 1e-8,verbose = 0,gnd = gnd,
                                         initialize_fp = 0,num_iters = iters * 1000,
                                         learning_rate = .01,solver_type = 'logistic')
                times[2,idx_exp,iters] = time_now.perf_counter() - start_time;
                w_rec = w[:opts['n_rec'],:].T.flatten()
                slopes[2,idx_exp,iters] = (gnd @ w_rec) / (gnd@gnd) 
                rmse[2,idx_exp,iters] = np.sqrt(np.mean( (w_rec-gnd)**2 ))
                rmedse[2,idx_exp,iters] = np.sqrt(np.median( (w_rec-gnd)**2 ))
                cor_p[2,idx_exp,iters] = pearsonr(w_rec,gnd)[0]
                cor_s[2,idx_exp,iters] = spearmanr(w_rec,gnd)[0]
                print('\t \t \t Pytorch,logistic, CPU. Cor %.3f' %(cor_p[2,idx_exp,iters]))
                if time_now.perf_counter() - start_time > time_limit:
                    counter[2] =1
            except:
                print('\t \t \t Pytorch,logistic, CPU terminated due to timeout error')
                counter[2] = 1
        else:
            print('\t \t \t Pytorch,logistic, CPU skipped due to timeout error')
    

        #%
        start_time = time_now.perf_counter()
        if counter[4] == 0:
            try:
                  
                w  = solve_pytorch_gpu(r[:-1,:],r[1:,:],u_in = None,alph = 0.1,
                                         l2 = 1e-8,verbose = 0,gnd = gnd, #1e-7 for short
                                         initialize_fp =1,num_iters = iters*1000,
                                         learning_rate = .01,solver_type = 'l2')
                times[4,idx_exp,iters] = time_now.perf_counter() - start_time;
                w_rec = w[:opts['n_rec'],:].T.flatten()
                slopes[4,idx_exp,iters] = (gnd @ w_rec) / (gnd@gnd)
                rmse[4,idx_exp,iters] = np.sqrt(np.mean( (w_rec-gnd)**2 ))
                rmedse[4,idx_exp,iters] = np.sqrt(np.median( (w_rec-gnd)**2 ))
                cor_p[4,idx_exp,iters] = pearsonr(w_rec,gnd)[0]
                cor_s[4,idx_exp,iters] = spearmanr(w_rec,gnd)[0]
                print('\t \t \t Pytorch,l2,GPU. Cor %.3f' %(cor_p[4,idx_exp,iters]))
                if time_now.perf_counter() - start_time > time_limit:
                    counter[4] =1
            except:
                print('\t \t \t Pytorch,l2,GPU terminated due to timeout error')
                counter[4] = 1
        else:
            print('\t \t \t Pytorch,l2,GPU skipped due to timeout error')
        
        
        #%
        start_time = time_now.perf_counter()
        if counter[6] == 0:
            try:
                 
                w = fit_FORCE(r,None,alph = 0.1,
                                lam = 100,g_in = 2,verbose = 0,
                                initialize_fp = 1,num_iters = 5*iters,
                                gnd = gnd,solver_type = 'currents')
                times[6,idx_exp,iters] = time_now.perf_counter() - start_time;
                w_rec = w[:opts['n_rec'],:].T.flatten()
                slopes[6,idx_exp,iters] = (gnd @ w_rec) / (gnd@gnd)
                rmse[6,idx_exp,iters] = np.sqrt(np.mean( (w_rec-gnd)**2 ))
                rmedse[6,idx_exp,iters] = np.sqrt(np.median( (w_rec-gnd)**2 ))
                cor_p[6,idx_exp,iters] = pearsonr(w_rec,gnd)[0]
                cor_s[6,idx_exp,iters] = spearmanr(w_rec,gnd)[0]
                print('\t \t \t Force,currents. Cor %.3f' %(cor_p[6,idx_exp,iters]))
                if time_now.perf_counter() - start_time > time_limit:
                    counter[6] =1
            except:
                print('\t \t \t Force,currents terminated due to timeout error')
                counter[6] = 1
        else:
            print('\t \t \t Force,currents skipped due to timeout error')
        
        
        #%
        start_time = time_now.perf_counter()
        if counter[7] == 0:
            try:
                  
                w = fit_FORCE(r,None,alph = 0.1,
                                lam = 100,g_in = 2,verbose = 0,
                                initialize_fp = 1,num_iters = 5*iters,
                                gnd = gnd,solver_type = 'firing_rates')
                times[7,idx_exp,iters] = time_now.perf_counter() - start_time;
                w_rec = w[:opts['n_rec'],:].T.flatten()
                slopes[7,idx_exp,iters] = (gnd @ w_rec) / (gnd@gnd)
                rmse[7,idx_exp,iters] = np.sqrt(np.mean( (w_rec-gnd)**2 ))
                rmedse[7,idx_exp,iters] = np.sqrt(np.median( (w_rec-gnd)**2 ))
                cor_p[7,idx_exp,iters] = pearsonr(w_rec,gnd)[0]
                cor_s[7,idx_exp,iters] = spearmanr(w_rec,gnd)[0]
                print('\t \t \t Force, firing rates. Cor %.3f' %(cor_p[7,idx_exp,iters]))
                if time_now.perf_counter() - start_time > time_limit:
                    counter[7] =1
            except:
                print('\t \t \t Force, firing rates terminated due to timeout error')
                counter[7] = 1
        else:
            print('\t \t \t Force, firing rates skipped due to timeout error')
            
        if iters == 0:
            counter = np.zeros(num_algs)


np.savez('experiment8_results.npz',slopes = slopes,
         rmse = rmse, rmedse = rmedse,cor_p = cor_p,cor_s = cor_s,
         times = times)
