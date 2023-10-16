#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 14:48:40 2022

@author: dinc
"""
import numpy as np
import cvxpy as cp
import multiprocessing
from joblib import Parallel, delayed
from scipy.stats import pearsonr
from numpy.linalg import solve
from scipy.linalg import solve as solve2
from utils import solve_single_problem

import time as time_now

class CoRNN():
    
    
    def __init__(self,opts = None):
        self.opts = self.get_defaults(opts);
        
        
    def get_defaults(self,opts = None):
        if opts is None:
            opts = {};
        
        if 'n_iter' not in opts:
            opts['n_iter'] = 100
        if 'lambda_reg' not in opts:
            opts['lambda_reg'] = 1e-4
        if 'parallel' not in opts:
            opts['parallel'] = 0
        if 'verbose' not in opts:
            opts['verbose'] = False
        if 'gamma_error_tol' not in opts:
            opts['gamma_error_tol'] = 1e-4
        opts['n_iter'] = np.ceil(opts['n_iter']).astype(int);
            
        return opts
    
    def fit(self,R=None,R_out = None,U=None):
    
        n_rec = R.shape[1]
        if U is not None:
            n_in = U.shape[1]
        else:
            n_in = 0
        
        W_rec = np.zeros([n_rec,n_rec])
        if n_in > 0:
            W_in  = np.zeros([n_rec,n_in])
        
        
        if self.opts['parallel'] == 0:
            for iters in range(n_rec):
                if n_in > 0:
                    [W_rec[iters,:],W_in[iters,:]] = solve_single_problem(R,R_out,U,iters,self.opts)
                else:
                    [W_rec[iters,:],_] = solve_single_problem(R,R_out,None,iters,self.opts)
                if self.opts['verbose'] == True:
                    print("Node %d finished." %iters)
                    
        else:
            if self.opts['num_cores'] == None:
                num_cores = multiprocessing.cpu_count()
            else:
                num_cores = self.opts['num_cores']
            opts = self.opts
                
            (result) = Parallel(n_jobs=num_cores)(delayed(solve_single_problem)(R,R_out,U,iters,opts) for iters in range(n_rec))
            for i in range(n_rec):
                result_temp = result[i]
                W_rec[i,:] = np.array(result_temp[0])
                if n_in>0:
                    W_in[i,:] = np.array(result_temp[1])

            
        if n_in >0:
            return W_rec,W_in
        else:
            return W_rec,0
        

        
        
        
        
        
 
