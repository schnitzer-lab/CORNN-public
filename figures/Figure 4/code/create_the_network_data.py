#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 10 11:48:03 2023

@author: dinc
"""


import numpy as np
from scipy.stats import pearsonr
from scipy.stats import spearmanr
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

num_trials  = 50;
n_in        = 1;
n_rec       = 5000;
T           = 1000;
width       = 20;
gaus_int    = 500;

r_in_all =np.zeros([T*num_trials,n_rec])
r_out_all =np.zeros([T*num_trials,n_rec])
u_all =np.zeros([T*num_trials,n_in])
out_all =np.zeros([T*num_trials,n_in])
    
model = Model(n_in,n_rec,n_in)
model = torch.load('model_0.pt')

for i in range(num_trials):
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
    out_all[i*T:(i+1)*T,:] = out;
        
    print(i)

np.savez('model0_input_noise0dot1_100trialdata.npz',r_in_all = r_in_all,
         r_out_all = r_out_all, u_all = u_all,out_all=out_all)




