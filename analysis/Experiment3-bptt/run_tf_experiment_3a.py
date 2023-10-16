#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 13:32:18 2022

@author: dinc
"""

import numpy as np
from scipy.stats import pearsonr
from rnn_class import RNN
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import random
import time as time_now
from joblib import Parallel, delayed
import multiprocessing


def run_pytorch_tf(r,u,tf_ratio,gnd,loss_type = 'l2',initialize_fp = 1,num_iters = 3000,optim_type = 'sgd'):
    
    # Define the network
    class Model(nn.Module):
        def __init__(self, input_dims, hidden_dims,tf_ratio = 0, alpha=0.9, device="cpu"):
            super(Model, self).__init__()
            self.input_dims = input_dims
            self.hidden_dims = hidden_dims
            self.alpha = torch.tensor(alpha).to(device)
            self.device = device
            self.tf_ratio = tf_ratio
    
            # initialize weights
            self.W_in = nn.Parameter(torch.randn(hidden_dims,input_dims))
            self.W_rec = nn.Parameter(torch.randn(hidden_dims, hidden_dims))
            nn.init.xavier_uniform_(self.W_in)
            nn.init.xavier_uniform_(self.W_rec)
    
            # set W_rec diagonal to zero
            self.W_rec.data = self.W_rec.data * (1 - torch.eye(hidden_dims))
    
            # move everything to device
            self.to(device)
        
        def forward(self, u, h, h_all):
            
            """
            The shape of u is (seq_len,input_dims)
            The shape of h is (hidden_dims), initialization of hidden dims
            The shape of hidden_states is (seq_len, hidden_dims)
            """
            T = u.shape[0]
            n_rec = h.shape[0]
            
            hidden_states = torch.zeros([T+1,n_rec]).to(self.device)
            hidden_states[0,:] = h.flatten();
            teacher_forcing_ratio = self.tf_ratio
            
            for t in range(T):
                use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
                if use_teacher_forcing:
                    h_from_prev = (1 - self.alpha) * h_all[t,:]
                    h = h_from_prev + self.alpha * torch.tanh(self.W_rec @ h_all[t,:] +self.W_in @ u[t,:] )
                    hidden_states[t+1,:] = h
                else:
                    h_from_prev = (1 - self.alpha) * h
                    h = h_from_prev + self.alpha * torch.tanh(self.W_rec @ h +self.W_in @ u[t,:] )
                    hidden_states[t+1,:] = h
                       
            return hidden_states
    
    
   
    
    # instantiate the model
    model = Model( opts['n_in'],opts['n_rec'],tf_ratio)
    #model.W_rec.data = torch.tensor(w_rec,dtype=torch.float32);
    h_in = torch.tensor(r[0,:], dtype=torch.float32).to(model.device)
    r = torch.tensor(r, dtype=torch.float32).to(model.device)
    u_in = torch.tensor(u, dtype=torch.float32).to(model.device)
    
    if initialize_fp:
        r_in = r[:-1,:];
        r_out = r[1:,:];
        d = (r_out - (1-model.alpha)*r_in)/model.alpha
        d[d<=-1+1e-6] = -1 + 1e-6;
        d[d>=1-1e-6]  = 1-1e-6;
        x = torch.cat((r_in,u_in),1);
        T_data = x.shape[0]
        n_tot = x.shape[1]
        reg_term = 1e-5 * T_data  * np.diag(np.ones(n_tot))
        A = x.T @ x + torch.tensor(reg_term);
        A = torch.tensor(A, dtype=torch.float32).to(model.device)
        
        Ainv = torch.linalg.inv(A);
        
        Xp   = Ainv @ x.T;
        z    = torch.arctanh(d);
        theta_fp = Xp @ z;
        theta_fp = theta_fp.cpu().detach().numpy().T
        theta_fp = torch.tensor(theta_fp, dtype=torch.float32)
        n_rec = opts['n_rec']
        model.W_rec.data = theta_fp[:,:n_rec];
        model.W_in.data = theta_fp[:,n_rec:];
        model.to(model.device)
        
        weights = model.W_rec.data
        prd = weights.to('cpu').detach().numpy().flatten()

    
    
    # define the multi target logistic regression loss function
    if loss_type == 'l2':
        criterion = nn.MSELoss()
    else:
        criterion = nn.BCELoss()
    
    # define the optimizer: adam with weight decay
    if optim_type == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9,weight_decay=1e-8)
    else:
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-8)
    


    # train the model
    for epoch in range(num_iters):
        # forward
        r_out = model(u_in,h_in,r)
        
        # compute the loss
        if loss_type == 'l2':
            loss = criterion(r_out.flatten(), r.flatten())
        else:
            loss = criterion( (r_out.flatten()+1)/2, (r.flatten() +1)/2 )
        # backward
        loss.backward()
        
        # update the weights
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        # zero the gradients
        optimizer.zero_grad()
        # print the loss 
        model.W_rec.data = model.W_rec.data * (1 - torch.eye(n_rec).to(model.device))

        
    
    
    # get weights
    weights = model.W_rec.data
    prd = weights.to('cpu').detach().numpy().flatten()
    return pearsonr(gnd,prd)[0]


# set the parameters values


T_data = 100;

num_exp = 10;
tf_ratios = [0,0.25,0.5,0.75,1,0,0.25,0.5,0.75,1]
num_tr = len(tf_ratios)

accuracy_rep = np.zeros([2,num_exp,2,num_tr+1]) + np.nan
times =        np.zeros([2,num_exp,2,num_tr])+ np.nan
for k in range(2):
    if k == 0:
        opts = {
            'g': 3,
            'n_rec': 100,
            'n_in': 1,
            'sigma_input': 1e-1,
            'sigma_conversion': 1e-5,
            'alpha': 0.9,
            'input_noise_type': 'Gaussian',
            'conversion_noise_type': 'Gaussian',
            'verbose': False,
            'lambda_reg': 1e-5,
            'num_cores': 36,
            'parallel': 1
        }
    else:
        opts = {
            'g': 3,
            'n_rec': 100,
            'n_in': 1,
            'sigma_input': 1e-2,
            'sigma_conversion': 1e-4,
            'alpha': 0.9,
            'input_noise_type': 'Gaussian',
            'conversion_noise_type': 'Gaussian',
            'verbose': False,
            'lambda_reg': 1e-5,
            'num_cores': 36,
            'parallel': 1
        }
    for i in range(num_exp):
        m1 = RNN(opts)
        u = np.zeros(T_data) + np.random.normal(0,0,T_data)
        u[0:100] = u[0:100] + 0.1
        u = np.atleast_2d(u).T # Note that input should always be 2D!
        r = m1.get_time_evolution(T = T_data,u = u) # Shape of r is (T_data, n_rec)
        r[r >= 1-1e-6] = 1-1e-6;
        r[r<=-1+1e-6] = -1+1e-6;
        gnd = m1.rnn['w_rec'].flatten()

        accuracy_rep[k,i,:,-1] = run_pytorch_tf(r,u,1,gnd,'l2',1,0)
        temp = time_now.localtime()
        current_time = time_now.strftime("%H:%M:%S", temp)
        print('%s: Experiment %d. Initial Accuracy %.3f.' %(current_time,i,accuracy_rep[k,i,0,-1]))
    
        for j in range(num_tr):
            for l in range(2):
                if l== 0:
                    optim_tp = 'sgd'
                else:
                    optim_tp = 'adam'
                start_time = time_now.perf_counter()
                if j<5:
                    accuracy_rep[k,i,l,j] = run_pytorch_tf(r,u,tf_ratios[j],gnd,'l2',optim_type=optim_tp)
                    temp = time_now.localtime()
                    current_time = time_now.strftime("%H:%M:%S", temp)
                    print('\t \t %s: Loss Type l2. TF ratio %.1f. Opt. type %s. Accuracy %.3f.' %(current_time,
                               tf_ratios[j],optim_tp,accuracy_rep[k,i,l,j]))
                else:
                    accuracy_rep[k,i,l,j] = run_pytorch_tf(r,u,tf_ratios[j],gnd,'bce',optim_type=optim_tp)
                    temp = time_now.localtime()
                    current_time = time_now.strftime("%H:%M:%S", temp)
                    print('\t \t %s: Loss Type BCE. TF ratio %.1f. Opt. type %s. Accuracy %.3f.' %(current_time,
                               tf_ratios[j],optim_tp,accuracy_rep[k,i,l,j]))
                times[k,i,l,j]=time_now.perf_counter() - start_time;

            
np.savez('experiment3a_results.npz',accuracy_rep = accuracy_rep, times = times)





