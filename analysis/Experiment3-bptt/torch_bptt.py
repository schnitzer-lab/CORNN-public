#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 13:32:18 2022

@author: dinc
"""

import numpy as np
from scipy.stats import pearsonr
from cornn_class import CoRNN
from rnn_class import RNN
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import random

# set the parameters values
opts = {
    'g': 3,
    'n_rec': 100,
    'n_in': 1,
    'sigma_input': 1e-1,
    'sigma_conversion': 1e-3,
    'alpha': 0.1,
    'input_noise_type': 'Gaussian',
    'conversion_noise_type': 'Gaussian',
    'verbose': False,
    'lambda_reg': 1e-5,
    'num_cores': 36,
    'parallel': 1
}

T_data = 1000;
m1 = RNN(opts)
u = np.zeros(T_data)
u[0:100] = 1; 
u = np.atleast_2d(u).T # Note that input should always be 2D!
r = m1.get_time_evolution(T = T_data,u = u) # Shape of r is (T_data, n_rec)

#%% Run CoRNN (earlier version reducing to logistic regression)

opts['parallel'] = 0;
opts['lambda_reg'] = 1e-5; # Play wth the regularization
m3 =  CoRNN(opts)
w_rec,w_in = m3.fit(r[:-1,:],r[1:,:],u) # supply r_in first and then r_out
gnd = m1.rnn['w_rec'].flatten()
prd_cornn = w_rec.flatten()

# plot the weights, ground truth and prediction by cornn, with scatter plot
plt.scatter(gnd,prd_cornn)
print(pearsonr(gnd,prd_cornn))
plt.show()

#%%
# Define the network
class Model(nn.Module):
    def __init__(self, input_dims, hidden_dims,tf_ratio = 0, alpha=0.1, device="cpu"):
        super(Model, self).__init__()
        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.alpha = alpha
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
        
        hidden_states = torch.zeros([T+1,n_rec])
        hidden_states[0,:] = h.flatten();
        teacher_forcing_ratio = self.tf_ratio;
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


h_in = torch.tensor(r[0,:], dtype=torch.float32)
r = torch.tensor(r, dtype=torch.float32)
u_in = torch.tensor(u, dtype=torch.float32)

# instantiate the model
model = Model( opts['n_in'],opts['n_rec'])
#model.W_rec.data = torch.tensor(w_rec,dtype=torch.float32);

# define the multi target logistic regression loss function
criterion = nn.MSELoss()

# define the optimizer: adam with weight decay
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-9)
n_rec = opts['n_rec']
print('By CoRNN: {}'.format(pearsonr(gnd,prd_cornn)))
# train the model
for epoch in range(15000):
    # forward
    r_out = model(u_in,h_in,r)

    # compute the loss
    loss = criterion(r_out.flatten(), r.flatten())
    
    # backward
    loss.backward()
    
    # update the weights
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
    optimizer.step()
    # zero the gradients
    optimizer.zero_grad()
    # print the loss 
    model.W_rec.data = model.W_rec.data * (1 - torch.eye(n_rec))
    weights = model.W_rec.data;
    prd = weights[:, :opts['n_rec']].detach().numpy().flatten()

    # print the loss and correlation at every 100th epoch
    if epoch % 100 == 0:
        print('epoch {}: loss {}, correlation {}'.format(epoch, loss.item(),pearsonr(gnd,prd)[0] ))


# get weights
weights = model.W_rec.data
prd = weights[:, :opts['n_rec']].detach().numpy().flatten()


# plot the weights, grnd truth and prediction, with scatter plot
fig, ax = plt.subplots()
ax.scatter(gnd, prd)
#ax.set_aspect('equal')
plt.show()

# print the correlation between ground truth and prediction
print('By regression: {}'.format(pearsonr(gnd,prd)))
# print the correlation between ground truth and prediction by cornn
print('By CoRNN: {}'.format(pearsonr(gnd,prd_cornn)))


