#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 17:52:09 2022

@author: dinc
"""

import numpy as np
import torch
import torch.nn as nn
from numpy.linalg import inv
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from rnn_class import RNN
from utils_admm import solve_corrn_admm_gpu
from utils_admm import solve_corrn_admm
from utils_kbit import get_data_from_kbit_task
from utils_kbit import generate_test_data
import time as time_now


#%%
num_alph   = 2;
num_ners   = 3;
num_exps   = 10;
ner_all    = np.array([3000,1000,100]);
alph_all   = np.array([0.5,0.9]);


for i in range(num_alph):
    alph = alph_all[i]
    for j in range(num_ners):
        n_rec = ner_all[j]
        opts = {'alpha' : alph,
                'n_in' : 3,
                'sigma_input' : 1e-2,
                'sigma_conversion' : 1e-3,
                'verbose' : 0,
                'n_rec' : n_rec}
        for k in range(10):
            data,rnn_gnd,trial_info =  get_data_from_kbit_task(\
                                        10000,alph,n_rec,
                                        k,opts)
            save_dataset = "kbit_data/model_kbit_%s_%s_%s_trialdata.npz" %(k,alph,n_rec);
            np.savez(save_dataset, 
                     data = data, rnn_gnd = rnn_gnd,
                     trial_info = trial_info,allow_pickle=True)
            temp = time_now.localtime()
            current_time = time_now.strftime("%H:%M:%S", temp)
            print('%s: Alpha %.1f. Ner %d. Exp %d. finished.'  %(current_time,alph,n_rec,k))
           
        
           
            
            

