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

f = np.load('experiment5_neural_scaling_results.npz')

times = f['times']
algs = ["CPU","GPU-32bit","GPU-64bit"]
alph_all   = np.array([0.5]);
ner_all    = np.array([100,1000,3000]);
num_exps   = 20;
num_ners  = np.logspace(np.log10(101),np.log10(3000),8).astype(int)

# The dimensions are [alg,num_exps,num_ners]
from matplotlib.cm import get_cmap
cmap = get_cmap('tab10')
color_all = [cmap(0),cmap(1),cmap(2)]


#%% Make scaling of time for CPU
slopes_sc = np.zeros(3)
count = 0
for num_ner in [0,2,1]:
    temp = times[num_ner,:,:]/60
    x = num_ners
    plt.errorbar(x,np.mean(temp,0),np.std(temp,0)/np.sqrt(num_exps),color = color_all[num_ner])
    plt.scatter(x,np.mean(temp,0),color = color_all[num_ner],lw = 3,label = algs[num_ner])
    slope, intercept, r_value, p_value, std_err = linregress(np.log(x)[-4:], np.log(np.mean(temp,0))[-4:])
    slopes_sc[count] = slope
    y = np.exp(intercept) * x**slope
    plt.plot(x,y,color = 'black',ls = '--',lw = 2,zorder=100)
    print(slope,r_value)
    count = count + 1

plt.xscale('log')
plt.yscale('log')
plt.xlabel('Number of neurons')
plt.ylabel('Time (min)')
plt.xticks([100,1000,3000],['100','1000','3000'])
plt.title(slopes_sc)
plt.legend(loc = 'upper left')
plt.ylim([1e-4,1e2])
plt.savefig('Figure S5A.pdf', bbox_inches='tight')
plt.show()

#%%
f = np.load('experiment5_neural_scaling_results_lowT.npz')

times = f['times']
algs = ["CPU","GPU-32bit","GPU-64bit"]
alph_all   = np.array([0.5]);
ner_all    = np.array([100,1000,3000]);
num_exps   = 20;
num_ners  = np.logspace(np.log10(101),np.log10(3000),8).astype(int)

# The dimensions are [alg,num_exps,num_ners]
from matplotlib.cm import get_cmap
cmap = get_cmap('tab10')
color_all = [cmap(0),cmap(1),cmap(2)]


#%% Make scaling of time for CPU
slopes_sc = np.zeros(3)
count = 0
for num_ner in [0,2,1]:
    temp = times[num_ner,:,:]/60
    x = num_ners
    plt.errorbar(x,np.mean(temp,0),np.std(temp,0)/np.sqrt(num_exps),color = color_all[num_ner])
    plt.scatter(x,np.mean(temp,0),color = color_all[num_ner],lw = 3,label = algs[num_ner])
    slope, intercept, r_value, p_value, std_err = linregress(np.log(x)[-4:], np.log(np.mean(temp,0))[-4:])
    slopes_sc[count] = slope
    y = np.exp(intercept) * x**slope
    plt.plot(x,y,color = 'black',ls = '--',lw = 2,zorder=100)
    print(slope,r_value)
    count = count + 1

plt.xscale('log')
plt.yscale('log')
plt.xlabel('Number of neurons')
plt.ylabel('Time (min)')
plt.xticks([100,1000,3000],['100','1000','3000'])
plt.title(slopes_sc)
plt.legend(loc = 'upper left')
plt.ylim([1e-4,1e2])
plt.savefig('Figure S5B.pdf', bbox_inches='tight')
plt.show()

