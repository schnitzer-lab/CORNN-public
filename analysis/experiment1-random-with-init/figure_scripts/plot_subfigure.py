#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 13:56:08 2023

@author: dinc
"""

import numpy as np
import matplotlib.pyplot as plt


plt.subplot(3,1,1)
f = np.load('experiment1a_results_fp.npz')
slopes = f['slopes']
rmse = f['rmse']
rmedse = f['rmedse']
cor_p = f['cor_p']
cor_s = f['cor_s']
times = f['times']

# 0 is CoRNN, CPU
# 1 is Pytorch, logistic, GPU
# 2 is Pytorch, logistic, CPU
# 3 is NT, weighted
# 4 is Pytorch, l2, GPU
# 5 is NT, logistic
# 6 is Force, currents
# 7 is Force, firing rates
# 8 is CoRNN, GPU



labels_all = ["Fixed point, weighted, CPU","Pytorch, CE, GPU", \
              "Pytorch, CE, CPU", "Newton, weighted", "Pytorch, l2, GPU", \
                  "Newton, CE", "Force, currents", "Force, firing rates", \
                  "Fixed point, weighted, GPU"]

for pick in [8,0,3,5,1,2,4,6,7]:

    temp = cor_p[pick,:,:].mean(0)
    temp_std = np.std(cor_p[pick,:,:],0) / np.sqrt(cor_p.shape[1])
    x = np.nanmean(times[pick,:,:],0)
    x_std = np.std(times[pick,:,:],0) / np.sqrt(cor_p.shape[1])
    plt.errorbar(temp,x,x_std,temp_std,label = labels_all[pick])
   

plt.yscale('log')

plt.xlim([0.84,1.01])
plt.xticks([0.9,1],fontsize=14)
plt.yticks([0.1,1,10,100],fontsize=14)
#plt.xlabel('Correlation between ground truth and inferred weights',fontsize = 15)    
plt.ylabel('Time (s)',fontsize = 15)




plt.subplot(3,1,2)
f = np.load('experiment1b_results_fp.npz')
slopes = f['slopes']
rmse = f['rmse']
rmedse = f['rmedse']
cor_p = f['cor_p']
cor_s = f['cor_s']
times = f['times']

# 0 is CoRNN, CPU
# 1 is Pytorch, logistic, GPU
# 2 is Pytorch, logistic, CPU
# 3 is NT, weighted
# 4 is Pytorch, l2, GPU
# 5 is NT, logistic
# 6 is Force, currents
# 7 is Force, firing rates
# 8 is CoRNN, GPU



labels_all = ["Fixed point, weighted, CPU","Pytorch, CE, GPU", \
              "Pytorch, CE, CPU", "Newton, weighted", "Pytorch, l2, GPU", \
                  "Newton, CE", "Force, currents", "Force, firing rates", \
                  "Fixed point, weighted, GPU"]

for pick in [8,0,3,5,1,2,4,6,7]:

    temp = cor_p[pick,:,:].mean(0)
    temp_std = np.std(cor_p[pick,:,:],0) / np.sqrt(cor_p.shape[1])
    x = np.nanmean(times[pick,:,:],0)
    x_std = np.std(times[pick,:,:],0) / np.sqrt(cor_p.shape[1])
    plt.errorbar(temp,x,x_std,temp_std,label = labels_all[pick])


plt.xlim([0.84,1.01])
plt.xticks([0.9,1])
plt.yticks([0.1,1,10,100],fontsize=14) 
plt.ylabel('Time (s)',fontsize = 15)
plt.yscale('log')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))



plt.subplot(3,1,3)
f = np.load('experiment1c_results_fp.npz')
slopes = f['slopes']
rmse = f['rmse']
rmedse = f['rmedse']
cor_p = f['cor_p']
cor_s = f['cor_s']
times = f['times']

# 0 is CoRNN, CPU
# 1 is Pytorch, logistic, GPU
# 2 is Pytorch, logistic, CPU
# 3 is NT, weighted
# 4 is Pytorch, l2, GPU
# 5 is NT, logistic
# 6 is Force, currents
# 7 is Force, firing rates
# 8 is CoRNN, GPU



labels_all = ["Fixed point, weighted, CPU","Pytorch, CE, GPU", \
              "Pytorch, CE, CPU", "Newton, weighted", "Pytorch, l2, GPU", \
                  "Newton, CE", "Force, currents", "Force, firing rates", \
                  "Fixed point, weighted, GPU"]

for pick in [8,0,3,5,1,2,4,6,7]:

    temp = cor_p[pick,:,:].mean(0)
    temp_std = np.std(cor_p[pick,:,:],0) / np.sqrt(cor_p.shape[1])
    x = np.nanmean(times[pick,:,:],0)
    x_std = np.std(times[pick,:,:],0) / np.sqrt(cor_p.shape[1])
    plt.errorbar(temp,x,x_std,temp_std,label = labels_all[pick])



plt.xlabel('Correlation between ground truth and inferred weights',fontsize = 15)    
plt.ylabel('Time (s)',fontsize = 15)
plt.yscale('log')
