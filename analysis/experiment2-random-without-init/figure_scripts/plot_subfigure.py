#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 13:56:08 2023

@author: dinc
"""

import numpy as np
import matplotlib.pyplot as plt



f = np.load('experiment2a_results.npz')
slopes = f['slopes']
rmse = f['rmse']
rmedse = f['rmedse']
cor_p = f['cor_p']
cor_s = f['cor_s']
times = f['times']

# 0 is CoRNN
# 1 is Pytorch, logistic, GPU
# 2 is Pytorch, l2, CPU
# 3 is Force, currents
# 4 is Force, firing rates
# 5 is FP
iter_list = np.array([1e-5,1e-4,1e-3,1e-2,1e-1,1,10,100])



labels_all = ["CoRNN","Pytorch, CE ", \
               "Pytorch, l2", \
                 "Force, currents", "Force, firing rates","FP initialization"]

for pick in [5,0,1,2,3,4]:
    f1 = 2/(1/cor_p[pick,:,:] + np.maximum(slopes[pick,:,:],1/slopes[pick,:,:]))
    temp = f1.mean(0)
    temp_std = np.std(f1,0) / np.sqrt(cor_p.shape[1])
    plt.errorbar(iter_list,temp,temp_std,label = labels_all[pick])
   

plt.xscale('log')

plt.legend()