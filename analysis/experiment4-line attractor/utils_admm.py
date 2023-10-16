#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 14:48:40 2022

@author: dinc
"""
import numpy as np
from scipy.linalg import solve
import re
import time as time_now
from numpy.linalg import inv
from scipy.stats import pearsonr
import multiprocessing
from joblib import Parallel, delayed
import torch
import torch.nn as nn
import torch.optim as optim


def approximate_newton(r_in,r_out,u_in = None, alph = 0.1, 
                l2 = 1e-4):
    
    start_time = time_now.perf_counter()
    T_data = r_in.shape[0]
    n_rec = r_in.shape[1]
    if u_in is not None:
        n_in = u_in.shape[1]
    else:
        n_in = 0
    
    # Get the inputs x and targets d
    if n_in >0:
        x = np.c_[r_in,u_in];
    else:
        x = r_in.copy()
    d = (r_out - (1-alph)*r_in)/alph
    d[d<=-1+1e-6] = -1 + 1e-6;
    d[d>=1-1e-6]  = 1-1e-6;
    
    
    # Scale the regularization and compute the fixed point
    l2  = l2*T_data;
    
    
    
    A = x.T @ x + (l2) * np.diag(np.ones(x.shape[1]));
    
    
    Ainv = inv(A);
    x = np.mean(x,0)
    Xp   = Ainv @ x;
    times = time_now.perf_counter() - start_time
    
    times = 10 * n_rec * times/4;
    
   
    return times

def solve_corrn_admm(r_in,r_out,u_in = None, alph = 0.1, 
                l2 = 1e-4, threshold = 1, rho = 100,
                verbose = 0,check_convergence = 0,mask = None,
                num_iters = 30,gnd = None,solver_type = 'weighted'):
    if verbose:
        temp = time_now.localtime()
        current_time = time_now.strftime("%H:%M:%S", temp)
        print('%s: Running CoRNN solver.' %(current_time))
    start_time = time_now.perf_counter()
    
    T_data = r_in.shape[0]
    n_rec = r_in.shape[1]
    if u_in is not None:
        n_in = u_in.shape[1]
    else:
        n_in = 0
    
    # Get the inputs x and targets d
    if n_in >0:
        x = np.c_[r_in,u_in];
    else:
        x = r_in.copy()
    d = (r_out - (1-alph)*r_in)/alph
    d[d<=-1+1e-6] = -1 + 1e-6;
    d[d>=1-1e-6]  = 1-1e-6;
    
    if verbose == 2:
        temp = time_now.localtime()
        current_time = time_now.strftime("%H:%M:%S", temp)
        print('\t %s: Performing initial inverse operations' %(current_time))
    
    # Scale the regularization and compute the fixed point
    l2  = l2*T_data;
    rho = l2 *rho;
    
    if mask is None:
        mask = np.r_[np.diag(np.ones(n_rec)),np.zeros([n_in,n_rec]) ]
        mask = (mask == 1)
    
    
    A = x.T @ x + (l2+rho) * np.diag(np.ones(x.shape[1]));
    
    
    Ainv = inv(A);
    Xp   = Ainv @ x.T;
    Xm  = Xp @ x; 
    z    = np.arctanh(d);
    theta_fp = Xp @ z;
    theta = theta_fp.copy();
    chi = theta_fp.copy();
    v     = - l2/rho * theta_fp;
    
    if ( (verbose == 2) & (gnd is not None) ) :
        prd = (chi[:n_rec,:].T).flatten();
        p_cor = pearsonr(gnd,prd)[0];
        slope = (gnd @ prd) / (gnd@gnd) 
        f1 = 2 / (1/p_cor + max(slope, 1/slope) )
        print('\t %s: Fixed point. f1 %.4f. cor: %.3f and slope %.3f. Starting iterations' %(current_time,f1,p_cor,slope))
        
    elif verbose == 2:
        temp = time_now.localtime()
        current_time = time_now.strftime("%H:%M:%S", temp)
        print('\t %s: Starting iterations' %(current_time))
        
    for idx in range(num_iters):
        # Compute predictions and prediction errors
        dhat      = np.tanh(x @ theta);
        if solver_type == 'weighted':
            E_pred    = (d-dhat) /(1-d**2); 
            scale = np.sum(abs(E_pred)>threshold,0);
            
            if verbose:
                per_not_used = np.mean(scale) / T_data;
                
            scale = T_data/(T_data-scale);
            
            E_pred[abs(E_pred)>threshold] = 0
            E_pred = E_pred * scale;
        elif solver_type == 'standard':
            E_pred    = (d-dhat);
            per_not_used = 0;
        elif solver_type == 'robust':
            E_pred    = (d-dhat) /(1-d**2);
            per_not_used = 0;
            E_pred[abs(E_pred)>threshold] = threshold * np.sign(E_pred[abs(E_pred)>threshold])
            
        
        
        
        
        # Perform the first primal variable update
        theta     = Xm @ theta + Xp @ E_pred + rho * Ainv @ (chi - v);
        
        # Perform the second primal variable update
        chi       = theta + v;
        chi[mask] = 0;

        # Perform the dual variable update
        v = v + theta - chi;

        conv = np.sqrt(n_rec) * np.sqrt(np.quantile((theta - chi ) **2 ,1-0.1/n_rec))
       
        
        if ( (verbose == 2) & (gnd is not None) ) :
            prd = (chi[:n_rec,:].T).flatten();
            p_cor = pearsonr(gnd,prd)[0];
        else:
            p_cor = np.nan;
        
            
            
        if (verbose == 2):
            temp = time_now.localtime()
            current_time = time_now.strftime("%H:%M:%S", temp)
            slope = (gnd @ prd) / (gnd@gnd) 
            
            f1 = 2 / ( max(slope, 1/slope)  + 1/p_cor)
            print('\t \t %s: Iteration %d finished. F1: %.4f. Correlation %.3f. Slope %.3f. Not used %.2f. Convergence %.6f.' \
                  %(current_time,idx + 1,f1,p_cor,slope,100*per_not_used,conv))
                
        if ((check_convergence>0) & (idx > 10)):
            if conv < 1e-3:
                break
    
    if verbose:
        temp = time_now.localtime()
        current_time = time_now.strftime("%H:%M:%S", temp)
        tt = time_now.perf_counter() - start_time
        print('%s: CoRNN solver finished in %.2f mins' %(current_time,tt/60))

    return chi

def solve_corrn_admm_gpu(r_in,r_out,u_in = None, alph = 0.1, 
                l2 = 1e-4, threshold = 1, rho = 100,
                verbose = 0,mask = None,float_type = '32bit',
                num_iters = 30,gnd = None,solver_type = 'weighted'):
    if verbose:
        temp = time_now.localtime()
        current_time = time_now.strftime("%H:%M:%S", temp)
        print('%s: Running CoRNN solver.' %(current_time))
    start_time = time_now.perf_counter()
    try:
        gpu_device = torch.device("cuda:0");
        a = torch.tensor(1,device = gpu_device);
    except:
        gpu_device = torch.device("mps");
    
    T_data = r_in.shape[0]
    n_rec = r_in.shape[1]
    if u_in is not None:
        n_in = u_in.shape[1]
    else:
        n_in = 0
    
    # Get the inputs x and targets d
    if n_in >0:
        x = np.c_[r_in,u_in];
    else:
        x = r_in.copy()
    d = (r_out - (1-alph)*r_in)/alph
    d[d<=-1+1e-6] = -1 + 1e-6;
    d[d>=1-1e-6]  = 1-1e-6;
    
    if float_type == '32bit':
        dtype = torch.float32
    else:
        dtype = torch.float64
    device = torch.device(gpu_device)
    n_tot = x.shape[1]
    
    x = torch.tensor(x,device=device, dtype=dtype)
    d = torch.tensor(d,device=device, dtype=dtype)
    
    if verbose == 2:
        temp = time_now.localtime()
        current_time = time_now.strftime("%H:%M:%S", temp)
        print('\t %s: Performing initial inverse operations' %(current_time))
        
        
    
    # Scale the regularization and compute the fixed point
    l2  = l2*T_data;
    rho = l2 *rho;
    
    
    
    if mask is None:
        mask = np.r_[np.diag(np.ones(n_rec)),np.zeros([n_in,n_rec]) ]
        mask = (mask == 1)
        
    reg_term = torch.tensor((l2+rho) * np.diag(np.ones(n_tot)),\
                            device=device, dtype=dtype)
    rho = torch.tensor(rho,device=device, dtype=dtype)

    A = x.T @ x + reg_term;
    
    try:
        Ainv = torch.linalg.inv(A);
    except:
        Ainv = torch.linalg.inv(A.to('cpu')).to(gpu_device);
    Xp   = Ainv @ x.T;
    Xm  = Xp @ x; 
    z    = torch.arctanh(d);
    theta_fp = Xp @ z;
    theta = theta_fp.clone();
    chi = theta_fp.clone();
    v     = - l2/rho * theta_fp.clone();
    
    # clean the gpu memory
    z = [];
    A = [];
    theta_fp =[];
    reg_term = [];
    
    if ( (verbose == 2) & (gnd is not None) ) :
        prd = (chi[:n_rec,:].T).flatten().cpu().numpy();
        p_cor = pearsonr(gnd,prd)[0];
        slope = (gnd @ prd) / (gnd@gnd) 
        f1 = 2 / (1/p_cor + max(slope, 1/slope) )
        print('\t %s: Fixed point. f1 %.4f. cor: %.3f and slope %.3f. Starting iterations' %(current_time,f1,p_cor,slope))
        
    elif verbose == 2:
        temp = time_now.localtime()
        current_time = time_now.strftime("%H:%M:%S", temp)
        print('\t %s: Starting iterations' %(current_time))
        
    for idx in range(num_iters):
        # Compute predictions and prediction errors
        dhat      = torch.tanh(x @ theta);
        if solver_type == 'weighted':
            E_pred    = (d-dhat) /(1-d**2); 
            per_not_used = torch.sum(abs(E_pred)>threshold) / (n_rec * T_data)
            E_pred[abs(E_pred)>threshold] = 0       
        elif solver_type == 'standard':
            E_pred    = (d-dhat);
            per_not_used = 0;
        elif solver_type == 'robust':
            E_pred    = (d-dhat) /(1-d**2);
            per_not_used = 0;
            E_pred[abs(E_pred)>threshold] = threshold * np.sign(E_pred[abs(E_pred)>threshold])
            
            
       
        
        
        
        
        # Perform the first primal variable update
        theta     = Xm @ theta + Xp @ E_pred + rho * Ainv @ (chi - v);
        
       
        # Perform the second primal variable update
        chi       = theta + v;
        chi[mask] = 0;

        # Perform the dual variable update
        v = v + theta - chi;

       
        
        if ( (verbose == 2) & (gnd is not None) ) :
            prd = (chi[:n_rec,:].T).flatten().cpu().numpy();
            p_cor = pearsonr(gnd,prd)[0];
        else:
            p_cor = np.nan;
        
        conv = np.sqrt(n_rec) * np.sqrt(np.quantile((theta.cpu().numpy() - chi.cpu().numpy() ) **2 ,1-0.1/n_rec))
       
            
            
        if (verbose == 2):
            temp = time_now.localtime()
            current_time = time_now.strftime("%H:%M:%S", temp)
            slope = (gnd @ prd) / (gnd@gnd) 
            rmse  = np.sqrt(np.mean( (prd-gnd)**2 ))
            f1 = 2 / ( max(slope, 1/slope) + 1/p_cor)
            print('\t \t %s: Iteration %d finished. F1: %.4f. Correlation %.3f. Slope %.3f. Not used %.2f. Convergence %.6f.' \
                  %(current_time,idx + 1,f1,p_cor,slope,100*per_not_used,conv))
                
    if verbose:
        temp = time_now.localtime()
        current_time = time_now.strftime("%H:%M:%S", temp)
        tt = time_now.perf_counter() - start_time
        print('%s: CoRNN solver finished in %.2f mins' %(current_time,tt/60))

    return chi.cpu().numpy()
