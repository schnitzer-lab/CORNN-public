#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 14:48:40 2022

@author: dinc
"""
import numpy as np
import multiprocessing
from joblib import Parallel, delayed
from scipy.stats import pearsonr
from numpy.linalg import solve
from scipy.linalg import solve as solve2

import time as time_now

class RNN():
    # Define some global variables that are mostly shared across all mice
    
    

    def __init__(self,opts=None):
        self.opts = self.get_defaults(opts);
        self.initialize_network()
        
        
    def get_defaults(self,opts = None):
        if opts is None:
            opts = {};
        
        if 'n_rec' not in opts:
            opts['n_rec'] = 30

        if 'n_in' not in opts:
            opts['n_in'] = 0
        
        if 'g' not in opts:
            opts['g'] = 3
            
        if 'sigma_input' not in opts:
            opts['sigma_input'] = 0
            
        if 'sigma_conversion' not in opts:
            opts['sigma_conversion'] = 0
            
        if 'alpha' not in opts:
            opts['alpha'] = 0.1
        
        if 'input_noise_type' not in opts:
            opts['input_noise_type'] = 'Gaussian'
            
        if 'conversion_noise_type' not in opts:
            opts['conversion_noise_type'] = 'Gaussian'
            
        return opts
        
    def initialize_network(self):
        n_rec = self.opts['n_rec']
        n_in  = self.opts['n_in']
        g     = self.opts['g']
        
        # Initialize the input weights
        nn={}
        if self.opts['n_in']>0:
            nn['w_in']=np.random.normal(0,g/np.sqrt(n_in),n_in*n_rec).reshape(n_rec,n_in)
            
        # Initialize the recurrent weights
        temp=np.random.normal(0,g/np.sqrt(n_rec),n_rec*n_rec).reshape(n_rec,n_rec)
        np.fill_diagonal(temp,0)
        nn['w_rec']=temp
        
        
        self.rnn = nn
        
            
    
    def get_time_evolution(self,r_in = None,u = None,T = None):
        if T is None:
            T = 1000
        if r_in is None:
            r_in = np.random.uniform(-1,1,self.opts['n_rec'])
        if u is None:
            if self.opts['n_in']>0:
                u = np.zeros([T,self.opts['n_in']])
                
        
        if self.opts['n_in']>0:
            w_in  = self.rnn['w_in']
            
        w_rec = self.rnn['w_rec']
        alpha = self.opts['alpha']
        sigma_input = self.opts['sigma_input']
        sigma_conversion = self.opts['sigma_conversion']
    
        n_rec = w_rec.shape[0]
        
        r = np.zeros([T+1,n_rec])
        
        r[0,:] = r_in
        for i in range(T):
            r_temp=r[i,:];
            
            if self.opts['input_noise_type'] == 'Gaussian':
                noise_input = np.random.normal(0,sigma_input,n_rec)
            elif self.opts['input_noise_type'] == 'Laplace':
                noise_input = np.random.laplace(0,sigma_input,n_rec)
            elif self.opts['input_noise_type'] == 'Poisson':
                noise_input = np.random.poisson(sigma_input,n_rec)
                
                
            if self.opts['n_in']>0:
                z =  np.dot(w_rec,r_temp) + np.dot(w_in,u[i,:]) \
                     + noise_input;
               
                
            else:
                z = np.dot(w_rec,r_temp)  + noise_input;
                
            
            if self.opts['conversion_noise_type'] == 'Gaussian':
                noise_conversion = np.random.normal(0,sigma_conversion,n_rec)
            elif self.opts['conversion_noise_type'] == 'Laplace':
                noise_conversion = np.random.laplace(0,sigma_conversion,n_rec)
            elif self.opts['conversion_noise_type'] == 'Poisson':
                noise_conversion = np.random.poisson(sigma_conversion,n_rec)
            
            temp_val = alpha*np.tanh(z) + (1-alpha)*r_temp + alpha*noise_conversion;
            
            # Newly added error could throw firing rates out of bounds, correct for it!
            #temp_val[temp_val >= 1] = 1;
            #temp_val[temp_val<=-1] = -1;
            
            r[i+1,:] = temp_val;
            
            
        
        return r
    
    def run_forward_propagation(self,r,u = None):
        """
        When inputting r, make sure you input it as r[:-1,:], as we want to compute the firing rates
        at time t+1, but r[T+1,:] is not defined, as r is defined between 0 and T, both inclusive!
        """
        T = r.shape[0]
        if u is not None and u.shape[0] != T:
            raise Exception('There is a missmatch between u and r dimensions!!!')
    
        if self.opts['n_in']>0:
            w_in  = self.rnn['w_in']
        w_rec = self.rnn['w_rec']
        alpha = self.opts['alpha']
        
        
        if self.opts['n_in']>0: 
            z = np.dot(r,w_rec.T) + np.dot(u,w_in.T) 
        else:
            z = np.dot(r,w_rec.T) 
            
        r_out = alpha*np.tanh(z) + (1-alpha)*r;
        return r_out
    
        
 
