#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 10 11:48:03 2023

@author: dinc
"""


import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import random
import time as time_now
from rnn_class import RNN
from utils_admm import solve_corrn_admm_gpu
from utils_admm import solve_corrn_admm
from sklearn.metrics import r2_score


# Define the network
class Model(nn.Module):
    def __init__(self, input_dims, hidden_dims,output_dims, alpha=0.1):
        super(Model, self).__init__()
        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.output_dims = output_dims
        self.alpha = alpha

        # initialize weights
        self.W_in = nn.Parameter(torch.randn(hidden_dims,input_dims))
        self.W_rec = nn.Parameter(torch.randn(hidden_dims, hidden_dims))
        self.W_out = nn.Parameter(torch.randn(output_dims,hidden_dims))
        nn.init.xavier_uniform_(self.W_in)
        nn.init.xavier_uniform_(self.W_rec)
        nn.init.xavier_uniform_(self.W_out)

        # set W_rec diagonal to zero
        self.W_rec.data = self.W_rec.data * (1 - torch.eye(hidden_dims))

    
    def forward(self, u, r,noise_in = 0.1, noise_con = 0):
        
        """
        Inputs
        The shape of u is (seq_len,input_dims)
        The shape of r is (hidden_dims), initialization of hidden dims
        
        Outputs
        The shape of o is (seq_len, output)
        The shape of hidden_states is (seq_len+1,hidden_dims)
        """
        T = u.shape[0]
        n_rec = r.shape[0]
        
        hidden_states = torch.zeros([T+1,n_rec])
        hidden_states[0,:] = r.flatten();
        x = torch.arctanh(r)
        o = torch.zeros([T,self.output_dims])
        for t in range(T):
            x = (1 - self.alpha) * x + self.alpha * (self.W_rec @ r + self.W_in @ u[t,:] \
                + torch.normal(torch.zeros(n_rec),noise_in) )
            r = torch.tanh(x) + torch.normal(torch.zeros(n_rec),noise_con) 
            hidden_states[t+1,:] = r
            o[t,:] = self.W_out @ r.flatten()
                
        return hidden_states,o


n_in        = 1;
n_rec       = 5000;
T           = 1000;
width       = 20;
gaus_int    = 500;

# 
n_sup_all   = np.logspace(2,np.log10(5000),8).astype(int)+1;
num_exps    = 10
num_trials  = np.array([30,50,100]).astype(int)

cor_weights_rec     = np.zeros([num_exps,num_trials.shape[0],n_sup_all.shape[0]])
cor_weights_in      = np.zeros([num_exps,num_trials.shape[0],n_sup_all.shape[0]])
r2_units            = np.zeros([num_exps,num_trials.shape[0],n_sup_all.shape[0]])
r2_units_dist       = np.zeros([num_exps,num_trials.shape[0],n_sup_all.shape[0]])
times               = np.zeros([num_exps,num_trials.shape[0],n_sup_all.shape[0]])
alph = 0.1 
for kk in range(num_exps):
    r_in_all =np.zeros([T*100,n_rec])
    r_out_all =np.zeros([T*100,n_rec])
    u_all =np.zeros([T*100,n_in])
    
    model = Model(n_in,n_rec,n_in)
    model = torch.load('model_%d.pt' %np.mod(kk,10))
    model.to('cpu')
    model.alpha = 0.1

    
    for i in range(100):
        u      = torch.zeros([T,n_in])
        u[0:100,0] = 1
        out_gt = torch.tensor(np.exp(-((np.arange(T) - gaus_int)**2)/(2*width**2)))
        
        r_init   = torch.rand(n_rec)-0.5
        # forward
        h,out = model(u,r_init)
        
        r = h.detach().numpy();
        out = out.detach().numpy();
        u = u.detach().numpy();
        gnd = model.W_rec.data.detach().numpy().flatten()
        u_all[i*T:(i+1)*T,:] = u;
        r_in_all[i*T:(i+1)*T,:] = r[:-1,:];
        r_out_all[i*T:(i+1)*T,:] = r[1:,:];
        
    
        
    for ll in range(num_trials.shape[0]):
        n_trial = num_trials[ll]
        r_in    = r_in_all[:n_trial * T,:]
        r_out   = r_out_all[:n_trial * T,:]
        u_in    = u_all[:n_trial * T,:]
        
        for ss in range(n_sup_all.shape[0]):
            n_sup = n_sup_all[ss]
            gnd = model.W_rec.data.detach().numpy()
            gnd = gnd[:n_sup,:n_sup].flatten()
            
            w = [];
            start_time = time_now.perf_counter();
            w = solve_corrn_admm(r_in[:,:n_sup],r_out[:,:n_sup],u_in = u_in, alph =alph , l2 = 1e-4, 
                                     threshold = 1, rho = 10,verbose = 0,num_iters = 50,
                                     gnd = gnd,solver_type = 'weighted')
            times[kk,ll,ss] = time_now.perf_counter()-start_time;
        
            w_rec_cornn = w[:n_sup,:].T
            w_in_cornn = w[n_sup:,:].T
            
            prd = w_rec_cornn.flatten()
            
            cor_weights_rec[kk,ll,ss] = pearsonr(gnd,prd)[0]
    
    
    
            gnd_in = model.W_in.data.detach().numpy()[:n_sup];
            prd_in = w_in_cornn.flatten()
    
            cor_weights_in[kk,ll,ss] = pearsonr(gnd_in.flatten(),prd_in)[0]
    
            
            
            
            r2_temp = np.zeros(3)
            r2_dist_temp= np.zeros(3)
            for ntest in range(3):
                opts = {};
                opts['n_rec'] = n_sup
                opts['n_in'] = 3
                opts['alpha'] = alph
                opts['verbose'] = False;
                opts['sigma_input'] = 0
                opts['sigma_conversion'] =0 
                m1 = RNN(opts)
                m1.rnn['w_rec'] = w_rec_cornn
                m1.rnn['w_in'] = w_in_cornn
                
                u      = torch.zeros([T,n_in])
                u[0:100,0] = 1
        
                
                r_init   = torch.rand(n_rec)-0.5
                # forward
                h,out = model(u,r_init,0)
                
                r_gt = h.detach().numpy();
                u = u.detach().numpy();
                #r = m1.get_time_evolution(T = u.shape[0], u =u)
                r_cornn = [];
                r_cornn = m1.get_time_evolution(T = u.shape[0], u =u,r_in = r_gt[0,:n_sup])
                
                
                
                r2_temp[ntest] = r2_score(r_gt[:,:100],r_cornn[:,:100],multioutput='uniform_average')
                
                
                opts = {};
                opts['n_rec'] = n_sup
                opts['n_in'] = 3
                opts['alpha'] = alph
                opts['verbose'] = False;
                opts['sigma_input'] = 0
                opts['sigma_conversion'] =0 
                m1 = RNN(opts)
                m1.rnn['w_rec'] = w_rec_cornn
                m1.rnn['w_in'] = w_in_cornn
                
                u      = torch.zeros([T,n_in])
                u[0:100,0] = 1
                u[400:410,0] = 10
        
                # forward
                h,out = model(u,r_init,0)
                
                r_gt = h.detach().numpy();
                u = u.detach().numpy();
                #r = m1.get_time_evolution(T = u.shape[0], u =u)
                r_cornn = [];
                r_cornn = m1.get_time_evolution(T = u.shape[0], u =u,r_in = r_gt[0,:n_sup])
                r2_dist_temp[ntest] = r2_score(r_gt[:,:100],r_cornn[:,:100],multioutput='uniform_average')
                
                
            r2_units[kk,ll,ss] = np.mean(r2_temp)
            r2_units_dist[kk,ll,ss] = np.mean(r2_dist_temp)
            
            temp = time_now.localtime()
            current_time = time_now.strftime("%H:%M:%S", temp)
            print('%s: Experiment %d. Num Trials %d. N_sup %d. Time: %.2f. Input %.2f. Rec %.2f. R2 %.2f and %.2f.' %(current_time, 
                                                         kk,n_trial,n_sup ,times[kk,ll,ss],cor_weights_in[kk,ll,ss]
                                                         , cor_weights_rec[kk,ll,ss],r2_units[kk,ll,ss],
                                                         r2_units_dist[kk,ll,ss]))

np.savez('subsampling_experiment.npz',r2_units = r2_units,
         cor_weights_in=cor_weights_in, times = times,
         cor_weights_rec=cor_weights_rec,r2_units_dist = r2_units_dist)
