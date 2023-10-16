#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 14:48:40 2022

@author: dinc
"""
import numpy as np
import multiprocessing
from joblib import Parallel, delayed
from scipy.stats import pearsonr
from numpy.linalg import solve
from scipy.linalg import solve as solve2
import re

import time as time_now

def solve_single_problem(r_in,r_tar,U,iters,opts):

    alph = opts['alpha']
    
    n_rec = r_in.shape[1]
    if U is not None:
        n_in = U.shape[1]
    else:
        n_in = 0
    
    w_rec = np.zeros(n_rec)
    if n_in>0:
        w_in = np.zeros(n_in)
    if r_tar.any() == None:
        r_tar = r_in[1:,iters]
        r_in = r_in[:-1,:]
    else:
        r_tar = r_tar[:,iters]
    
    d_tar = (r_tar - (1-alph)*r_in[:,iters] )/alph;
    d_tar [d_tar < -1] = -1
    d_tar [d_tar > 1] = 1
    
    ind = np.linspace(0,n_rec-1,n_rec).astype(int)
    ind = np.setdiff1d(ind,iters)
    r_in = r_in[:,ind]
    
    w_rec[ind],w_in = solve_newton_descent(U,r_in,d_tar,opts)
    
    return w_rec,w_in
    
def solve_newton_descent(u,r_in,d_tar,opts):
    
    reg = opts['lambda_reg']
    n_iter = opts['n_iter']

    if u is not None:
        n_in = u.shape[1]
    else:
        n_in = 0
    
    n_rec = r_in.shape[1]

    if u is not None:
        n_in = u.shape[1]
        beta = np.zeros([n_rec + n_in])
    else:
        beta = np.zeros(n_rec)
        n_in = 0
    Gamma = 1
    for idx in range(n_iter):
       
        
        grad = compute_gradient(u,r_in,d_tar,beta,reg)
        
        hes  = compute_hessian(u,r_in,d_tar,beta,reg)
        del_nt = -solve2(hes, grad,assume_a='pos')
        kappa = 1
        gamma = 0.4;
        chi = 0.6;
        count = 0
        stop_flag = 0
        
        while(   compute_loss_function_full(u,r_in,d_tar,beta + kappa * del_nt,reg) >=  \
              compute_loss_function_full(u,r_in,d_tar,beta,reg) + gamma * kappa* (grad @ del_nt)  ):
    
            kappa = chi * kappa
            count = count + 1
            if count == 10:
                if opts['verbose'] == True:
                    print('BT stuck, Gamma: %.7f' %Gamma)
                kappa = 1;
                break
            
        delta_gd = kappa * del_nt
        
        beta = beta + delta_gd
        if stop_flag == 0:
            Gamma = np.sqrt(-grad @ del_nt)
        else:
            Gamma = 0
        #print("Iteration %d finished." %idx)
        
        
        if Gamma < opts['gamma_error_tol']:
            if opts['verbose'] == True:
                print("Iteration number %d. Gamma %.5f. Kappa %.2f" %(idx,Gamma,kappa))
            break
        
            
    if n_in > 0:
        w_rec = beta[:n_rec]
        w_in  = beta[n_rec:]
    else:
        w_rec = beta
        w_in  = 0
    
    return w_rec,w_in


def compute_loss_function(u,r,d,beta):
    
    
    if u is not None:
        
        n_rec = r.shape[1]
        w_rec = beta[:n_rec]
        w_in  = beta[n_rec:]
        
        dhat = np.tanh( u @ w_in + r @ w_rec )
        dhat[dhat>=1] = 1-1e-15
        dhat[dhat<=-1] = -1+1e-15
        
        loss = - np.mean( 0.5*(d+1)* np.log(0.5*(dhat+1)) +  0.5*(1-d)* np.log(0.5*(1-dhat))   )
    else:
        dhat = np.tanh( r @ beta )
        #dhat[dhat>=1] = 1-1e-6
        #dhat[dhat<=-1] = -1+1e-6
        loss = - np.mean( 0.5*(d+1)* np.log(0.5*(dhat+1)) +  0.5*(1-d)* np.log(0.5*(1-dhat))   )
    
    return loss


def compute_loss_function_full(u,r,d,beta,reg):
    
    
    if u is not None:
        
        n_rec = r.shape[1]
        w_rec = beta[:n_rec]
        w_in  = beta[n_rec:]
        
        dhat = np.tanh( u @ w_in + r @ w_rec )
        dhat[dhat>=1] = 1-1e-6
        dhat[dhat<=-1] = -1+1e-6
        
        loss = - np.mean( 0.5*(d+1)* np.log(0.5*(dhat+1)) +  0.5*(1-d)* np.log(0.5*(1-dhat))   ) + reg * np.sum(beta**2)/2
    else:
        dhat = np.tanh( r @ beta )
        dhat[dhat>=1] = 1-1e-15
        dhat[dhat<=-1] = -1+1e-15
        loss = - np.mean( 0.5*(d+1)* np.log(0.5*(dhat+1)) +  0.5*(1-d)* np.log(0.5*(1-dhat))   ) + reg * np.sum(beta**2)/2
    
    return loss



def compute_gradient(u,r,d,beta,reg):
    
    
    
    if u is not None:
        n_in = u.shape[1]
    else:
        n_in = 0
        
    if n_in > 0:
        n_rec = r.shape[1]
        gradient = np.zeros(n_in + n_rec)
        n_rec = r.shape[1]
        w_rec = beta[:n_rec]
        w_in  = beta[n_rec:]
        
        dhat = np.tanh( u @ w_in + r @ w_rec )
        gradient[:n_rec] = np.mean( (dhat-d)[:,None]*r,axis=0 ) + reg * w_rec
        gradient[n_rec:] = np.mean( (dhat-d)[:,None]*u, axis =0 ) + reg * w_in
    else:
        dhat = np.tanh( r @ beta )
        gradient =  np.mean( (dhat-d)[:,None]*r,axis=0 ) + reg * beta
    
    return gradient


def compute_hessian(u,r,d,beta,reg):
    
    
    
    if u is not None:
        n_in = u.shape[1]
    else:
        n_in = 0
        
    if n_in > 0:
        n_rec = r.shape[1]
        n_rec = r.shape[1]
        w_rec = beta[:n_rec]
        w_in  = beta[n_rec:]
        
        dhat = np.tanh( u @ w_in + r @ w_rec )
        
        temp = np.sqrt(1-dhat**2)
        
        r = r * temp[:,None];
        u = u * temp[:,None];
        
        H       = np.zeros([n_rec+n_in,n_rec+n_in])
        
        H_rec   = r.T @ r/r.shape[0]+ reg * np.diag(np.ones(n_rec))
        H_in    = u.T @ u/r.shape[0]+ reg * np.diag(np.ones(n_in))
        H_cross = u.T @ r/r.shape[0]
        
        H[:n_rec,:n_rec] = H_rec
        H[:n_rec,n_rec:] = H_cross.T
        H[n_rec:,:n_rec] = H_cross
        H[n_rec:,n_rec:] = H_in

       
    else:
        n_rec = r.shape[1]
        dhat = np.tanh( r @ w_rec )
        temp = np.sqrt(1-dhat**2)
        
        r = r * temp[:,None];
        H   = r.T @ r /r.shape[0] + reg * np.diag(np.ones(n_rec))
        
    
    return H

def rnd(number):
    n = abs(number)
    if n < 1:
        # Find the first non-zero digit.
        # We want 3 digits, starting at that location.
        s = f'{n:.99f}'
        index = re.search('[1-9]', s).start()
        return s[:index + 3]
    else:
        # We want 2 digits after decimal point.
        return str(round(n, 2))
        

        
        
        
        
        
 
