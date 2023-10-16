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

f = np.load('correlated_noise_experiment.npz')

cor_weights_rec = f['cor_weights_rec']
cor_weights_in = f['cor_weights_in']
r2_units_dist = f['r2_units_dist']
times           = f['times']

n_sup_all   = np.array([1e-3,1e-2,0.05,0.1,0.3,0.5,1,10])
num_exps    = 5
num_trials  = np.array([30,50,100]).astype(int)

# The dimensions are [num_exps,num_trials,n_sup]
from matplotlib.cm import get_cmap
cmap = get_cmap('tab10')
color_all = [cmap(0),cmap(1),cmap(2)]


#%% Make figure N1a

n_ch = 4
n_sup_all = n_sup_all[:n_ch]
for i in range(num_trials.shape[0]):
    temp = r2_units_dist[:,1,i,:]
    temp = temp[:,:n_ch]
    plt.errorbar(n_sup_all,np.mean(temp,0),np.std(temp,0)/np.sqrt(num_exps),color = color_all[i],label = num_trials[i])
    #plt.scatter(n_sup_all,np.mean(temp,0),color = color_all[i],lw = 3)
    temp = r2_units_dist[:,0,i,:]
    temp = temp[:,:n_ch]
    plt.errorbar(n_sup_all,np.mean(temp,0),np.std(temp,0)/np.sqrt(num_exps),
                 color = color_all[i],ls = '--')
    #plt.scatter(n_sup_all,np.mean(temp,0),color = color_all[i],lw = 3)
    


#plt.xlim([0.01,1.1])
plt.xscale('log')
plt.xlabel('Noise s.d.')
plt.ylabel('Accuracy ($R^2$) of output in a novel distraction trial')
plt.legend()
#plt.xticks([0.01,0.1,1],['1%','10%','100%'])
plt.savefig('Figure_N1.pdf', bbox_inches='tight')
plt.show()


