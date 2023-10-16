#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 10:12:07 2022

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
import matplotlib.patches as patches


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


#%% Get network data
f = np.load('model0_input_noise0dot1_100trialdata.npz')
r_in_all = f['r_in_all']
r_out_all = f['r_out_all']
u_all     = f['u_all']
out_all   = f['out_all']

n_in        = 1;
n_rec       = 5000;    
T = 700

model = Model(n_in,n_rec,n_in)
model = torch.load('model_0.pt')
r_init   = torch.rand(n_rec)-0.5
out_gt = np.exp(-((np.arange(T) - 500)**2)/(2*20**2))


u      = torch.zeros([T,n_in])
u[0:100,0] = 1

u_no = u.detach().numpy().copy();

# forward
h,out = model(u,r_init,0.1)
r_or_no = h.detach().numpy();
out_or_no = out.detach().numpy()

u[200:210,0] = 0.5
# forward
h,out = model(u,r_init,0.1)
r_or_yes = h.detach().numpy();
out_or_yes = out.detach().numpy().copy()


u_yes    = u.detach().numpy();
r_init   = r_init.detach().numpy()
W_out_nw = model.W_out.data.detach().numpy()
W_rec_nw = model.W_rec.data.detach().numpy()
W_in_nw  = model.W_in.data.detach().numpy()

#%% Plot Figure 7B, Original network outputs without distractor
_,ax = plt.subplots(figsize = (9,3))

ax.add_patch(patches.Rectangle((0, 0), 50, 10, color='black', alpha=0.3,zorder=100))
ax.add_patch(patches.Rectangle((460, 0), 80, 10, color='black', alpha=0.3,zorder=100))
plt.imshow(r_or_no[:,:500].T,aspect = 'auto',cmap = 'jet',vmin = -0.5,vmax = 0.5,interpolation = None)
plt.xlabel('Time (ms)')
plt.ylabel('Neuron ID')
plt.colorbar()
plt.savefig('Figure 7B1.pdf', bbox_inches='tight')
plt.show()

_,ax = plt.subplots(figsize = (9,1.5))
plt.plot(u_no,color = 'black',ls = '--',label = 'Input')
plt.plot(out_gt,color = 'black',label = 'Target Output')
plt.plot(out_or_no,color = 'blue',label = 'Network Output')
plt.xlabel('Time (ms)')
plt.ylabel('Input/output (a.u.)')
plt.savefig('Figure 7B2.pdf', bbox_inches='tight')
plt.show()


#%% Plot Figure 7C, Original network outputs with distractor
_,ax = plt.subplots(figsize = (9,3))

ax.add_patch(patches.Rectangle((0, 0), 50, 10, color='black', alpha=0.3,zorder=100))
ax.add_patch(patches.Rectangle((460, 0), 80, 10, color='black', alpha=0.3,zorder=100))
ax.add_patch(patches.Rectangle((200, 0), 10, 10, color='black', alpha=0.3,zorder=100))
plt.imshow(r_or_yes[:,:500].T,aspect = 'auto',cmap = 'jet',vmin = -0.5,vmax = 0.5,interpolation = None)
plt.xlabel('Time (ms)')
plt.ylabel('Neuron ID')
plt.colorbar()
plt.savefig('Figure 7C1.pdf', bbox_inches='tight')
plt.show()

_,ax = plt.subplots(figsize = (9,1.5))
plt.plot(u_yes,color = 'black',ls = '--',label = 'Input')
plt.plot(out_gt,color = 'black',label = 'Target Output')
plt.plot(out_or_yes,color = 'blue',label = 'Network Output')
plt.xlabel('Time (ms)')
plt.ylabel('Input/output (a.u.)')
plt.legend()
plt.savefig('Figure 7C2.pdf', bbox_inches='tight')
plt.show()



#%% Train the networks for rest of the figures

f = np.load('model0_input_noise0dot1_100trialdata.npz')
r_in_all = f['r_in_all']
r_out_all = f['r_out_all']
u_all     = f['u_all']
out_all   = f['out_all']

n_sup = 5000;
nt=100
alph = 0.1
r_in_train = r_in_all[:1000*nt,:n_sup]
r_out_train = r_out_all[:1000*nt,:n_sup]
u_in_train  = u_all[:1000*nt,:]
out_train   = out_all[:1000*nt,:]
gnd = model.W_rec.data.detach().numpy()
gnd = gnd[:n_sup,:n_sup].flatten()

w = solve_corrn_admm(r_in_train,r_out_train,u_in = u_in_train, alph =alph , l2 = 1e-5, 
                         threshold = 1, rho = 100,verbose = 2,num_iters = 50,
                         gnd = gnd,solver_type = 'weighted')

a = np.linalg.lstsq(r_in_train,out_train.flatten())
w_out = a[0]

np.savez('fig7_full5000_cornn_data.npz',w=w,w_out=w_out)


n_sup = 500;
nt=100
alph = 0.1
r_in_train = r_in_all[:1000*nt,:n_sup]
r_out_train = r_out_all[:1000*nt,:n_sup]
u_in_train  = u_all[:1000*nt,:]
out_train   = out_all[:1000*nt,:]
gnd = model.W_rec.data.detach().numpy()
gnd = gnd[:n_sup,:n_sup].flatten()

w = solve_corrn_admm(r_in_train,r_out_train,u_in = u_in_train, alph =alph , l2 = 1e-5, 
                         threshold = 1, rho = 100,verbose = 2,num_iters = 30,
                         gnd = gnd,solver_type = 'weighted')

a = np.linalg.lstsq(r_in_train,out_train.flatten())
w_out = a[0]

np.savez('fig7_sub500_cornn_data.npz',w=w,w_out=w_out)


n_sup = 500;
nt=100
alph = 0.1 + np.random.normal(0,0.01,n_sup)
r_in_train = r_in_all[:1000*nt,:n_sup]
r_out_train = r_out_all[:1000*nt,:n_sup]
u_in_train  = u_all[:1000*nt,:]
out_train   = out_all[:1000*nt,:]
gnd = model.W_rec.data.detach().numpy()
gnd = gnd[:n_sup,:n_sup].flatten()

w = solve_corrn_admm(r_in_train,r_out_train,u_in = u_in_train, alph =alph , l2 = 1e-5, 
                         threshold = 1, rho = 100,verbose = 2,num_iters = 30,
                         gnd = gnd,solver_type = 'weighted')

a = np.linalg.lstsq(r_in_train,out_train.flatten())
w_out = a[0]

np.savez('fig7_sub500_10percent_time_scale_variation_cornn_data.npz',w=w,w_out=w_out,alph = alph)



