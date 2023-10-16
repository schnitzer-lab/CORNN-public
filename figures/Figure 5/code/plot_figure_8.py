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

f = np.load('subsampling_experiment.npz')

cor_weights_rec = f['cor_weights_rec']
cor_weights_in = f['cor_weights_in']
r2_units_dist = f['r2_units_dist']
times           = f['times']

n_sup_all   = np.logspace(2,np.log10(5000),8).astype(int)+1;
num_exps    = 10
num_trials  = np.array([30,50,100]).astype(int)

# The dimensions are [num_exps,num_trials,n_sup]
from matplotlib.cm import get_cmap
cmap = get_cmap('tab10')
color_all = [cmap(0),cmap(1),cmap(2)]


#%% Make figure 8a


for i in range(num_trials.shape[0]):
    temp = r2_units_dist[:,i,:]
    plt.errorbar(n_sup_all/5000,np.mean(temp,0),np.std(temp,0)/np.sqrt(num_exps),color = color_all[i])
    plt.scatter(n_sup_all/5000,np.mean(temp,0),color = color_all[i],lw = 3)


plt.xlim([0.01,1.1])
plt.xscale('log')
plt.xlabel('Subsampling percentage out of 5000 neurons')
plt.ylabel('Accuracy ($R^2$) of output in a novel distraction trial')
plt.xticks([0.01,0.1,1],['1%','10%','100%'])
plt.savefig('Figure 8A.pdf', bbox_inches='tight')
plt.show()

#%% Make figure 8b

for i in range(num_trials.shape[0]):
    temp = cor_weights_in[:,i,:]
    plt.errorbar(n_sup_all/5000,np.mean(temp,0),np.std(temp,0)/np.sqrt(num_exps),color = color_all[i])
    plt.scatter(n_sup_all/5000,np.mean(temp,0),color = color_all[i],lw = 3)


plt.xlim([0.01,1.1])
plt.xscale('log')
plt.xlabel('Subsampling percentage out of 5000 neurons')
plt.ylabel('Reconstruction accuracy (r) of $W_{in}$')
plt.xticks([0.01,0.1,1],['1%','10%','100%'])
plt.savefig('Figure 8B.pdf', bbox_inches='tight')
plt.show()
#%% Make figure 8c

for i in range(num_trials.shape[0]):
    temp = cor_weights_rec[:,i,:]
    plt.errorbar(n_sup_all/5000,np.mean(temp,0),np.std(temp,0)/np.sqrt(num_exps),color = color_all[i])
    plt.scatter(n_sup_all/5000,np.mean(temp,0),color = color_all[i],lw = 3)


plt.xlim([0.01,1.1])
plt.xscale('log')
plt.xlabel('Subsampling percentage out of 5000 neurons')
plt.ylabel('Reconstruction accuracy (r) of $W_{rec}$')
plt.xticks([0.01,0.1,1],['1%','10%','100%'])
plt.savefig('Figure 8C.pdf', bbox_inches='tight')
plt.show()
#%% Make figure 6c, scaling of time
for i in range(num_trials.shape[0]):
    temp = times[:,i,:]/60
    x = n_sup_all
    plt.errorbar(x,np.mean(temp,0),np.std(temp,0)/np.sqrt(num_exps),color = color_all[i])
    plt.scatter(x,np.mean(temp,0),color = color_all[i],lw = 3,label = num_trials[i])
    slope, intercept, r_value, p_value, std_err = linregress(np.log(x)[-4:], np.log(np.mean(temp,0))[-4:])
    y = np.exp(intercept) * x**slope
    plt.plot(x,y,color = 'black',ls = '--',lw = 2,zorder=100)
    print(slope,r_value)

plt.xscale('log')
plt.yscale('log')
plt.xlabel('Number of neurons')
plt.ylabel('Time (min)')
plt.xticks([100,1000,5000],['100','1000','5000'])
plt.legend(title='Number of trials')
plt.savefig('Figure 8D.pdf', bbox_inches='tight')
plt.show()




