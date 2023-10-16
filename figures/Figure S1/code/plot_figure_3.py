#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 17:52:09 2022

@author: dinc
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy
from numpy.matlib import repmat 
from scipy.stats import wilcoxon

f = np.load('experiment3b_results.npz')

acc = f['accuracy_rep']
times = f['times']

# The size is (noise_dist (2), num_exp, type_optim (1), num_tf)

from matplotlib.cm import get_cmap
cmap = get_cmap('tab10')
color_all = [cmap(0),cmap(1),cmap(2)]
tf_ratios = [0,0.25,0.5,0.75,1]

#%% Make figure 3a

temp = acc[0,:,0,[10,0,5,1,6,2,7,3,8,4,9]].T

times_temp = times[0,:,0,[0,5,1,6,2,7,3,8,4,9]]

fig, ax = plt.subplots(1, 1, figsize=(8,2))
# Set line width of axes
ax.spines["top"].set_linewidth(0)
ax.spines["bottom"].set_linewidth(2)
ax.spines["right"].set_linewidth(0)
ax.spines["left"].set_linewidth(2)
ax.tick_params(axis="both", which="both", bottom=True, top=False,
               labelbottom=True, left=True, right=False,
               labelleft=True,direction='out',length=10,width=2.0,pad=8,labelsize=10)

plt.boxplot(temp,positions = [0,0.8,1.2,1.8,2.2,2.8,3.2,3.8,4.2,4.8,5.2],widths = 0.2, 
            flierprops={'marker': 'o', 'markersize': 1})

plt.xticks([0,1,2,3,4,5],['Fixed-point',' 0 \n L2   CE','0.25 \n L2   CE','0.5 \n L2   CE',
                                                  '0.75 \n L2   CE','1 \n L2   CE'])
plt.xlabel('Teacher-forcing ratio')
plt.ylabel('Reconstruction accuracy (r) of $W_{rec}$')
plt.title('Runtime per run is %.1f $\pm$ %.1f seconds' %(np.mean(times_temp),np.std(times_temp)))
plt.savefig('Figure 3A.pdf', bbox_inches='tight')



#%% Make figure 3b, scaling of output accuracy for CPU

temp = acc[1,:,0,[10,0,5,1,6,2,7,3,8,4,9]].T

times_temp = times[1,:,0,[0,5,1,6,2,7,3,8,4,9]]

fig, ax = plt.subplots(1, 1, figsize=(8,2))
# Set line width of axes
ax.spines["top"].set_linewidth(0)
ax.spines["bottom"].set_linewidth(2)
ax.spines["right"].set_linewidth(0)
ax.spines["left"].set_linewidth(2)
ax.tick_params(axis="both", which="both", bottom=True, top=False,
               labelbottom=True, left=True, right=False,
               labelleft=True,direction='out',length=10,width=2.0,pad=8,labelsize=10)

plt.boxplot(temp,positions = [0,0.8,1.2,1.8,2.2,2.8,3.2,3.8,4.2,4.8,5.2],widths = 0.2, 
            flierprops={'marker': 'o', 'markersize': 1})

plt.xticks([0,1,2,3,4,5],['Fixed-point',' 0 \n L2   CE','0.25 \n L2   CE','0.5 \n L2   CE',
                                                  '0.75 \n L2   CE','1 \n L2   CE'])
plt.xlabel('Teacher-forcing ratio')
plt.ylabel('Reconstruction accuracy (r) of $W_{rec}$')
plt.title('Runtime per run is %.1f $\pm$ %.1f seconds' %(np.mean(times_temp),np.std(times_temp)))
plt.savefig('Figure 3B.pdf', bbox_inches='tight')
