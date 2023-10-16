#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 14 22:17:09 2023

@author: dinc
"""


import numpy as np
import matplotlib.pyplot as plt


file_names = ['FigureS1a.pdf','FigureS1b.pdf']

count = 0
for file in ['./experiment2a_results.npz','./experiment2b_results.npz']:
    f = np.load(file)
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



    labels_all = ["CoRNN","Gradient descent - CE ", \
                   "Gradient descent - L2", \
                     "Force - currents", "Force - firing rates","Fixed point (at init.)"]

    fig, ax = plt.subplots(1, 1, figsize=(7,3.5))
    # Set line width of axes
    ax.spines["top"].set_linewidth(0)
    ax.spines["bottom"].set_linewidth(2)
    ax.spines["right"].set_linewidth(0)
    ax.spines["left"].set_linewidth(2)

    # Turn on/off axes, ticks, change width, labelsize
    ax.tick_params(axis="both", which="both", bottom=True, top=False,
                   labelbottom=True, left=True, right=False,
                   labelleft=True,direction='out',length=10,width=2.0,pad=8,labelsize=20)

    for pick in [5,3,4]:
        f1 = 2/(1/cor_p[pick,:,:] + np.maximum(slopes[pick,:,:],1/slopes[pick,:,:]))
        temp = f1.mean(0)
        temp_std = np.std(f1,0) / np.sqrt(cor_p.shape[1])
        plt.errorbar(iter_list,temp,temp_std,label = labels_all[pick], lw=2)


    plt.xlabel("Regularization level", size=20)
    plt.ylabel("Accuracy", size=20)
    plt.xscale('log')
    plt.legend(fontsize = 12, loc=0)
    plt.tight_layout()
    plt.savefig(file_names[count])
    count = count + 1