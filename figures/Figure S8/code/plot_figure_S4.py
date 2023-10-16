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

f = np.load('experiment5_results.npz')

corrs = f['cor_p']
times = f['times']
out_r2 = f['out_r2']
algs = ["CPU","32bit","64bit"]
alph_all   = np.array([0.5]);
ner_all    = np.array([100,1000,3000]);
num_exps   = 20;
num_trials = np.array([25,50,100,200,400,1000,200*10,200*30])

# The dimensions are [alg,alpha,num_ner,num_exps,num_trials]
from matplotlib.cm import get_cmap
cmap = get_cmap('tab10')
color_all = [cmap(0),cmap(1),cmap(2)]


#%% Make scaling of output accuracy for CPU


for num_ner in range(3):
    temp = out_r2[0,0,num_ner,:,:]
    plt.errorbar(num_trials,np.mean(temp,0),np.std(temp,0)/np.sqrt(num_exps),color = color_all[num_ner])
    plt.scatter(num_trials,np.mean(temp,0),color = color_all[num_ner],lw = 3)

plt.xscale('log')
plt.xlabel('Number of trials')
plt.ylabel('Accuracy ($R^2$) of output')
plt.savefig('Figure S4A.pdf', bbox_inches='tight')
plt.show()

#%% Make scaling of recurrent weight accuracy for CPU

for num_ner in range(3):
    temp = corrs[0,0,num_ner,:,:]
    plt.errorbar(num_trials,np.mean(temp,0),np.std(temp,0)/np.sqrt(num_exps),color = color_all[num_ner])
    plt.scatter(num_trials,np.mean(temp,0),color = color_all[num_ner],lw = 3)

plt.xscale('log')
plt.xlabel('Number of trials')
plt.ylabel('Reconstruction accuracy (r) of $W_{rec}$')
plt.savefig('Figure S4B.pdf', bbox_inches='tight')
plt.show()

#%% Make scaling of time for CPU
slopes_sc = np.zeros(3)
for num_ner in range(3):
    temp = times[0,0,num_ner,:,:]/60
    x = num_trials
    plt.errorbar(x,np.mean(temp,0),np.std(temp,0)/np.sqrt(num_exps),color = color_all[num_ner])
    plt.scatter(x,np.mean(temp,0),color = color_all[num_ner],lw = 3,label = ner_all[num_ner])
    slope, intercept, r_value, p_value, std_err = linregress(np.log(x)[-4:], np.log(np.mean(temp,0))[-4:])
    slopes_sc[num_ner] = slope
    y = np.exp(intercept) * x**slope
    plt.plot(x,y,color = 'black',ls = '--',lw = 2,zorder=100)
    print(slope,r_value)

plt.xscale('log')
plt.yscale('log')
plt.xlabel('Number of trials')
plt.ylabel('Time (min)')
plt.xticks([100,200,1000,6000],['100','One day \n of data','1000','A Month \n of data'])
plt.title(slopes_sc)
plt.legend(title='Number of neurons')
plt.ylim([1e-4,1e2])
plt.savefig('Figure S4C.pdf', bbox_inches='tight')
plt.show()



#%% Make scaling of output accuracy for GPU 64bit


for num_ner in range(3):
    temp = out_r2[2,0,num_ner,:,:]
    plt.errorbar(num_trials,np.mean(temp,0),np.std(temp,0)/np.sqrt(num_exps),color = color_all[num_ner])
    plt.scatter(num_trials,np.mean(temp,0),color = color_all[num_ner],lw = 3)

plt.xscale('log')
plt.xlabel('Number of trials')
plt.ylabel('Accuracy ($R^2$) of output')
plt.savefig('Figure S4D.pdf', bbox_inches='tight')
plt.show()

#%% Make scaling of recurrent weight accuracy for GPU 64bit

for num_ner in range(3):
    temp = corrs[2,0,num_ner,:,:]
    plt.errorbar(num_trials,np.mean(temp,0),np.std(temp,0)/np.sqrt(num_exps),color = color_all[num_ner])
    plt.scatter(num_trials,np.mean(temp,0),color = color_all[num_ner],lw = 3)

plt.xscale('log')
plt.xlabel('Number of trials')
plt.ylabel('Reconstruction accuracy (r) of $W_{rec}$')
plt.savefig('Figure S4E.pdf', bbox_inches='tight')
plt.show()

#%% Make scaling of time for GPU 64bit
slopes_sc = np.zeros(3)
for num_ner in range(3):
    temp = times[2,0,num_ner,:,:]/60
    x = num_trials
    plt.errorbar(x,np.mean(temp,0),np.std(temp,0)/np.sqrt(num_exps),color = color_all[num_ner])
    plt.scatter(x,np.mean(temp,0),color = color_all[num_ner],lw = 3,label = ner_all[num_ner])
    if num_ner < 1:
        slope, intercept, r_value, p_value, std_err = linregress(np.log(x)[-4:], np.log(np.mean(temp,0))[-4:])
    elif num_ner < 2:
        slope, intercept, r_value, p_value, std_err = linregress(np.log(x)[-5:-1], np.log(np.mean(temp,0))[-5:-1])
    else:
        slope, intercept, r_value, p_value, std_err = linregress(np.log(x)[-5:-2], np.log(np.mean(temp,0))[-5:-2])
    y = np.exp(intercept) * x**slope
    slopes_sc[num_ner] = slope
    plt.plot(x,y,color = 'black',ls = '--',lw = 2,zorder=100)
    print(slope,r_value)

plt.xscale('log')
plt.yscale('log')
plt.xlabel('Number of trials')
plt.ylabel('Time (min)')
plt.xticks([100,200,1000,6000],['100','One day \n of data','1000','A Month \n of data'])
plt.legend(title='Number of neurons')
plt.title(slopes_sc)
plt.ylim([1e-4,1e2])
plt.savefig('Figure S4F.pdf', bbox_inches='tight')
plt.show()

#%% Make scaling of output accuracy for GPU 32bit


for num_ner in range(3):
    temp = out_r2[1,0,num_ner,:,:]
    plt.errorbar(num_trials,np.mean(temp,0),np.std(temp,0)/np.sqrt(num_exps),color = color_all[num_ner])
    plt.scatter(num_trials,np.mean(temp,0),color = color_all[num_ner],lw = 3)

plt.xscale('log')
plt.xlabel('Number of trials')
plt.ylabel('Accuracy ($R^2$) of output')
plt.savefig('Figure S4G.pdf', bbox_inches='tight')
plt.show()

#%% Make scaling of recurrent weight accuracy for GPU 32bit

for num_ner in range(3):
    temp = corrs[1,0,num_ner,:,:]
    plt.errorbar(num_trials,np.mean(temp,0),np.std(temp,0)/np.sqrt(num_exps),color = color_all[num_ner])
    plt.scatter(num_trials,np.mean(temp,0),color = color_all[num_ner],lw = 3)

plt.xscale('log')
plt.xlabel('Number of trials')
plt.ylabel('Reconstruction accuracy (r) of $W_{rec}$')
plt.savefig('Figure S4H.pdf', bbox_inches='tight')
plt.show()

#%% Make scaling of time for GPU 32bit
slopes_sc = np.zeros(3)
for num_ner in range(3):
    temp = times[1,0,num_ner,:,:]/60
    x = num_trials
    plt.errorbar(x,np.mean(temp,0),np.std(temp,0)/np.sqrt(num_exps),color = color_all[num_ner])
    plt.scatter(x,np.mean(temp,0),color = color_all[num_ner],lw = 3,label = ner_all[num_ner])
    if num_ner == 0:
        slope, intercept, r_value, p_value, std_err = linregress(np.log(x)[-4:], np.log(np.mean(temp,0))[-4:])

    else:
        slope, intercept, r_value, p_value, std_err = linregress(np.log(x)[-4:-2], np.log(np.mean(temp,0))[-4:-2])
    y = np.exp(intercept) * x**slope
    slopes_sc[num_ner] = slope
    plt.plot(x,y,color = 'black',ls = '--',lw = 2,zorder=100)
    print(slope,r_value)

plt.xscale('log')
plt.yscale('log')
plt.xlabel('Number of trials')
plt.ylabel('Time (min)')
plt.xticks([100,200,1000,6000],['100','One day \n of data','1000','A Month \n of data'])
plt.legend(title='Number of neurons')
plt.title(slopes_sc)
plt.ylim([1e-4,1e2])
plt.savefig('Figure S4I.pdf', bbox_inches='tight')
plt.show()



