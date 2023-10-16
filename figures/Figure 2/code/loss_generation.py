#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 14:48:40 2022

@author: dinc
"""
import numpy as np
import matplotlib.pyplot as plt

# define the target within [0,1]
target = 0.2

x_target = np.log(target/(1-target))


x = np.linspace(-10 + x_target, 10 + x_target, 100)
z = 1/(1 + np.exp(-x))



l2_loss = (z-target)**2 # L2 loss
ce_loss = target * np.log(target) + (1-target) * np.log(1-target) - \
    (target * np.log(z) + (1-target) * np.log(1-z) )
# Rescale ce loss
ce_loss = ce_loss * (1-target) * target * 2

plt.plot(x,ce_loss)
plt.plot(x,l2_loss)

 

#%%

# define the target within [0,1]
target = 0.5
thresh = 0.2

x_target = np.log(target/(1-target))


x = np.linspace(-3 + x_target, 3 + x_target, 1000)
z = 1/(1 + np.exp(-x))



l2_loss = (z-target)**2 # L2 loss
ce_loss = target * np.log(target) + (1-target) * np.log(1-target) - \
    (target * np.log(z) + (1-target) * np.log(1-z) )

E_pred = (z - target) / ( target * (1- target) )

ind = np.where(abs(E_pred) > thresh )
ce_loss_rb = ce_loss.copy()
ce_loss_rb[ind] = thresh *  abs(x[ind]-x_target)

# Rescale ce loss
ce_loss_rb = ce_loss_rb * (1-target) * target * 2
ce_loss = ce_loss * (1-target) * target * 2

plt.plot(x,ce_loss)
plt.plot(x,l2_loss)
plt.plot(x,ce_loss_rb)









 
