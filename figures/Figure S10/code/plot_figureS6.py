#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 17:52:09 2022

@author: dinc
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.stats import linregress

f = np.load('timescales_experiment.npz')

cor_weights_rec = f['cor_weights_rec'][:,:,1:]
cor_weights_in = f['cor_weights_in'][:,:,1:]
r2_units_dist = f['r2_units_dist'][:,:,1:]
times           = f['times'][:,:,1:]

t_sup_all   = np.array([0,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.2,0.3])[1:]
num_exps    = 10
num_trials  = np.array([30,50,100]).astype(int)

# The dimensions are [num_exps,num_trials,t_sup]
from matplotlib.cm import get_cmap
cmap = get_cmap('tab10')
color_all = [cmap(0),cmap(1),cmap(2)]


#%% Make figure 8a


for i in range(num_trials.shape[0]):
    temp = r2_units_dist[:,i,:]
    plt.errorbar(t_sup_all,np.mean(temp,0),np.std(temp,0)/np.sqrt(num_exps),color = color_all[i])
    plt.scatter(t_sup_all,np.mean(temp,0),color = color_all[i],lw = 3)


plt.xlim([0.009,1.1])
plt.xscale('log')
plt.xlabel('Uncertainty (s.d.) in time scales')
plt.ylabel('Accuracy ($R^2$) of output in a novel distraction trial')
plt.xticks([0.01,0.1,1],['0.01','0.1','1'])
plt.savefig('Figure S6A.pdf', bbox_inches='tight')
plt.show()

#%% Make figure 8b

for i in range(num_trials.shape[0]):
    temp = cor_weights_in[:,i,:]
    plt.errorbar(t_sup_all,np.mean(temp,0),np.std(temp,0)/np.sqrt(num_exps),color = color_all[i])
    plt.scatter(t_sup_all,np.mean(temp,0),color = color_all[i],lw = 3)


plt.xlim([0.009,1.1])
plt.xscale('log')
plt.xlabel('Uncertainty (s.d.) in time scales')
plt.ylabel('Reconstruction accuracy (r) of $W_{in}$')
plt.xticks([0.01,0.1,1],['0.01','0.1','1'])
plt.savefig('Figure S6B.pdf', bbox_inches='tight')
plt.show()
#%% Make figure 8c

for i in range(num_trials.shape[0]):
    temp = cor_weights_rec[:,i,:]
    plt.errorbar(t_sup_all,np.mean(temp,0),np.std(temp,0)/np.sqrt(num_exps),color = color_all[i])
    plt.scatter(t_sup_all,np.mean(temp,0),color = color_all[i],lw = 3)


plt.xlim([0.009,1.1])
plt.xscale('log')
plt.xlabel('Uncertainty (s.d.) in time scales')
plt.ylabel('Reconstruction accuracy (r) of $W_{rec}$')
plt.xticks([0.01,0.1,1],['0.01','0.1','1'])
plt.savefig('Figure S6C.pdf', bbox_inches='tight')
plt.show()


#%% Make figure 6c, scaling of time
for i in range(num_trials.shape[0]):
    temp = times[:,i,:]/60
    x = t_sup_all
    plt.errorbar(x,np.mean(temp,0),np.std(temp,0)/np.sqrt(num_exps),color = color_all[i])
    plt.scatter(x,np.mean(temp,0),color = color_all[i],lw = 3,label = num_trials[i])

plt.xscale('log')
plt.xlabel('Uncertainty (s.d.) in time scales')
plt.ylabel('Time (min)')
plt.legend(title='Number of trials')
plt.savefig('Figure S6D.pdf', bbox_inches='tight')


