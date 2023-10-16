#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 14:48:40 2022

@author: dinc
"""
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["OMP_NUM_THREADS"] = "5"
os.environ['OPENBLAS_NUM_THREADS'] = '5'
os.environ['MKL_NUM_THREADS'] = '5'

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


def solve_corrn(r_in,r_out,u_in = None, alph = 0.1, 
                l2 = 1e-4, threshold = 1, initialize_fp = 1,
                verbose = 0,check_convergence = 0,
                num_iters = 30,gnd = None,solver_type = 'weighted',exp_id = 0):

    temp = time_now.localtime()
    start_time = time_now.strftime("%H:%M:%S", temp)
    print('\t %s: Running CoRNN solver.' %(start_time))
    start_time = time_now.perf_counter()
    save_name = 'CORNN_CPU_exp_%d' %exp_id;

    times  = np.zeros(num_iters+1)
    slopes = np.zeros(num_iters+1)
    cors   = np.zeros(num_iters+1)

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
    n_tot = n_rec + n_in
    
    Ainv = inv(A);
    Xp   = Ainv @ x.T;
    Xm  = Xp @ x; 
    z    = np.arctanh(d);
    theta_fp = Xp @ z;
    if initialize_fp:
        theta = theta_fp;
    else:
        theta = np.random.normal(0,1/np.sqrt(n_rec),n_rec * n_tot).reshape(n_tot,n_rec)
        
    end_time = time_now.perf_counter()
    times[0] = end_time - start_time
    prd = (theta[:n_rec,:].T).flatten();
    cors[0]  = pearsonr(gnd,prd)[0];
    slopes[0]= (gnd @ prd) / (gnd@gnd)


    for idx in range(num_iters):

        start_time = time_now.perf_counter()
        theta_old = theta.copy()
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
        theta     = Xm @ theta + Xp @ E_pred;
        
       
        
        
        end_time = time_now.perf_counter()
        times[idx + 1] = end_time - start_time;
        
            
            
        prd = (theta[:n_rec,:].T).flatten();
        p_cor = pearsonr(gnd,prd)[0];
        cors[idx+1] = p_cor
        temp = time_now.localtime()
        current_time = time_now.strftime("%H:%M:%S", temp)
        slope = (gnd @ prd) / (gnd@gnd)
        slopes[idx+1] = slope
        if np.mod(idx,20) == 0:
            print('\t \t %s: Iteration %d finished.  Correlation %.3f. Slope %.3f. Time %.2f mins. ' \
                  %(current_time,idx + 1,p_cor,slope,np.sum(times)/60))
                    

    temp = time_now.localtime()
    current_time = time_now.strftime("%H:%M:%S", temp)
    print('\t %s: CoRNN CPU solver finished in %.3f mins.' %(current_time,np.sum(times)/60))
    np.savez(save_name,times = times,slopes = slopes, cors = cors)

    return theta


def solve_corrn_gpu(r_in,r_out,u_in = None, alph = 0.1,  float_type = '32bit',
                l2 = 1e-4, threshold = 0.2,initialize_fp = 1,
                verbose = 0,mask = None, check_convergence = 0,
                num_iters = 30,gnd = None,solver_type = 'weighted',exp_id = 0):
 

    temp = time_now.localtime()
    start_time = time_now.strftime("%H:%M:%S", temp)
    print('\t %s: Running CoRNN solver.' %(start_time))
    start_time = time_now.perf_counter()
    save_name = 'CORNN_GPU_exp_%d' %exp_id;

    times  = np.zeros(num_iters+1)
    slopes = np.zeros(num_iters+1)
    cors   = np.zeros(num_iters+1)

    try:
        gpu_device = torch.device("cuda:0");
        a = torch.tensor(1,device = gpu_device);
    except:
        gpu_device = torch.device("mps");
    num_iters = int(num_iters);
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
    n_tot = x.shape[1]
    
    x = torch.tensor(x,device=gpu_device, dtype=dtype)
    d = torch.tensor(d,device=gpu_device, dtype=dtype)
   
    # Scale the regularization and compute the fixed point
    l2  = l2*T_data;
    
    
    
    if mask is None:
        mask = np.r_[np.diag(np.ones(n_rec)),np.zeros([n_in,n_rec]) ]
        mask = (mask == 1)
        
    reg_term = torch.tensor((l2) * np.diag(np.ones(n_tot)),\
                            device=gpu_device, dtype=dtype)

    A = x.T @ x + reg_term;
    
    
    try:
        Ainv = torch.linalg.inv(A);
    except:
        Ainv = torch.linalg.inv(A.cpu()).to(gpu_device);
    Xp   = Ainv @ x.T;
    Xm  = Xp @ x; 
    z    = torch.arctanh(d);
    theta_fp = Xp @ z;
    if initialize_fp:
        theta = theta_fp;
    else:
        w_rec = np.random.normal(0,1/np.sqrt(n_rec),n_rec*n_tot).reshape(n_tot,n_rec)
        theta = torch.tensor(w_rec,device=gpu_device, dtype=dtype);
    

    
    end_time = time_now.perf_counter()
    times[0] = end_time - start_time
    prd = (theta[:n_rec,:].T).flatten().cpu().numpy();
    cors[0]  = pearsonr(gnd,prd)[0];
    slopes[0]= (gnd @ prd) / (gnd@gnd)

    temp = time_now.localtime()
    current_time = time_now.strftime("%H:%M:%S", temp)
   
    print('\t \t %s: FP initialized:  Correlation %.3f. Slope %.3f. ' \
          %(current_time,cors[0],slopes[0]))

    
    for idx in range(num_iters):
        # Compute predictions and prediction errors
        start_time = time_now.perf_counter()
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
        theta     = Xm @ theta + Xp @ E_pred

        end_time = time_now.perf_counter()
        times[idx + 1] = end_time - start_time;
       
        
        prd = (theta[:n_rec,:].T).flatten().cpu().numpy();
        p_cor = pearsonr(gnd,prd)[0];
        cors[idx+1] = p_cor
        temp = time_now.localtime()
        current_time = time_now.strftime("%H:%M:%S", temp)
        slope = (gnd @ prd) / (gnd@gnd)
        slopes[idx+1] = slope
        if np.mod(idx,10) == 0:
            print('\t \t %s: Iteration %d finished.  Correlation %.3f. Slope %.3f. Time %.2f mins.. ' \
                  %(current_time,idx + 1,p_cor,slope,np.sum(times)/60))
        
                
    
    
    temp = time_now.localtime()
    current_time = time_now.strftime("%H:%M:%S", temp)
    print('\t %s: CoRNN GPU solver finished in %.3f mins.' %(current_time,np.sum(times/60)))
    np.savez(save_name,times = times,slopes = slopes, cors = cors)

    return theta.cpu().numpy()

def solve_pytorch(r_in,r_out,u_in,alph = 0.1,
                         l2 = 1e-5,verbose = 0,
                         initialize_fp = 0,num_iters = 1e4,time_limit = 24,
                         threshold = 2,learning_rate = 0.001,
                         gnd = None,solver_type = 'logistic',exp_id = 0):
    temp = time_now.localtime()
    start_time = time_now.strftime("%H:%M:%S", temp)
    print('\t %s: Running Pytorch CPU solver, type_%s_fpinit_%d.' %(start_time,solver_type,initialize_fp))
    save_name = 'Pytorch_CPU_type_%s_fpinit_%d_exp_%d' %(solver_type,initialize_fp,exp_id);

    times  = np.zeros(1)
    slopes = np.zeros(1)
    cors   = np.zeros(1)

    start_time = time_now.perf_counter()
    
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
    if initialize_fp:
        T_data = x.shape[0]
        A = x.T @ x + l2 *T_data *1e4* np.diag(np.ones(x.shape[1]));
        
        
        Ainv = inv(A);
        Xp   = Ainv @ x.T;
        z    = np.arctanh(d);
        theta_fp = Xp @ z;

    
    # Define the network
    class Model(nn.Module):
        def __init__(self, n_rec, n_in):
            super(Model, self).__init__()
            self.n_rec = n_rec
            self.n_in = n_in
            self.linear = nn.Linear(n_rec + n_in, n_rec,bias=False)
            self.tanh = nn.Tanh()
        
        def forward(self, x_in):
            #x_in = torch.tensor(x_in, dtype=torch.float32)
            d_out = self.tanh(self.linear(x_in))
            return d_out
        
    d = torch.tensor(d, dtype=torch.float32)
    x = torch.tensor(x, dtype=torch.float32)
    model = Model(n_rec, n_in)
    # define the multi target logistic regression loss function
    if solver_type == 'logistic':
        criterion = nn.BCELoss()
    else:
        criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2)
    if initialize_fp:
        model.linear.weight.data = torch.tensor(theta_fp.T, dtype=torch.float32)

    end_time = time_now.perf_counter()
    times[0] = end_time - start_time
    weights = model.linear.weight
    prd = weights[:, :n_rec].detach().numpy().flatten()
    cors[0]  = pearsonr(gnd,prd)[0];
    slopes[0]= (gnd @ prd) / (gnd@gnd)
    count_time = 0
    for idx in range(num_iters):
        start_time = time_now.perf_counter()
        d_out = model(x)

        # compute the loss
        loss = criterion((1 + d_out)/2, (1 + d)/2)
        
        # backward
        loss.backward()
        
        
        # update the weights
        optimizer.step()
        # zero the gradients
        optimizer.zero_grad()
        
        end_time = time_now.perf_counter()
        count_time = count_time + end_time - start_time
        
        if np.mod(idx,1000) == 0:
            times = np.append(times,count_time);
            count_time = 0;
           
            weights = model.linear.weight
            prd = weights[:, :n_rec].detach().numpy().flatten()
            p_cor = pearsonr(gnd,prd)[0];
            cors= np.append(cors,p_cor)
            temp = time_now.localtime()
            current_time = time_now.strftime("%H:%M:%S", temp)
            slope = (gnd @ prd) / (gnd@gnd)
            slopes = np.append(slopes, slope)
            print('\t \t %s: Iteration %d finished.  Correlation %.3f. Slope %.3f. Time %.2f mins.  ' \
                  %(current_time,idx + 1,p_cor,slope,np.sum(times)/60))
    
    
        if (np.sum(times)>3600*time_limit):
            break
    temp = time_now.localtime()
    current_time = time_now.strftime("%H:%M:%S", temp)
    print('\t %s: Solver finished in %.3f mins.' %(current_time,np.sum(times)/60))
    np.savez(save_name,times = times,slopes = slopes, cors = cors)
    
    return model.linear.weight.detach().numpy().T



def solve_pytorch_gpu(r_in,r_out,u_in,alph = 0.1,
                         l2 = 1e-5,verbose = 0,
                         initialize_fp = 0,num_iters = 1e4,
                         threshold = 2,learning_rate = 0.001, time_limit = 24,
                         gnd = None,solver_type = 'logistic',exp_id = 0):
    temp = time_now.localtime()
    start_time = time_now.strftime("%H:%M:%S", temp)
    print('\t %s: Running Pytorch GPU solver, type_%s_fpinit_%d.' %(start_time,solver_type,initialize_fp))
    save_name = 'Pytorch_GPU_type_%s_fpinit_%d_exp_%d' %(solver_type,initialize_fp,exp_id);

    times  = np.zeros(1)
    slopes = np.zeros(1)
    cors   = np.zeros(1)

    start_time = time_now.perf_counter()
    
    n_rec = r_in.shape[1]
    if u_in is not None:
        n_in = u_in.shape[1]
    else:
        n_in = 0

    try:
        gpu_device = torch.device("cuda:0");
        a = torch.tensor(1,device = gpu_device);
    except:
        gpu_device = torch.device("mps");
    
    # Get the inputs x and targets d
    if n_in >0:
        x = np.c_[r_in,u_in];
    else:
        x = r_in.copy()
    d = (r_out - (1-alph)*r_in)/alph
    d[d<=-1+1e-6] = -1 + 1e-6;
    d[d>=1-1e-6]  = 1-1e-6;

    # Define the network
    class Model(nn.Module):
        def __init__(self, n_rec, n_in):
            super(Model, self).__init__()
            self.n_rec = n_rec
            self.n_in = n_in
            self.linear = nn.Linear(n_rec + n_in, n_rec,bias=False)
            self.tanh = nn.Tanh()
        
        def forward(self, x_in):
            #x_in = torch.tensor(x_in, dtype=torch.float32)
            d_out = self.tanh(self.linear(x_in))
            return d_out
    


    d = torch.tensor(d, dtype=torch.float32,device = gpu_device)
    x = torch.tensor(x, dtype=torch.float32,device = gpu_device)

    if initialize_fp:
        T_data = x.shape[0]
        n_tot = n_rec+ n_in
        reg_term = torch.tensor((l2) * T_data * 1e4 * np.diag(np.ones(n_tot)),\
                                device=gpu_device,dtype = torch.float32)

        A = x.T @ x + reg_term;
        
        try:
            Ainv = torch.linalg.inv(A);
        except:
            Ainv = torch.linalg.inv(A.cpu()).to(gpu_device);
        Xp   = Ainv @ x.T;
        z    = torch.arctanh(d);
        theta_fp = Xp @ z;
        theta_fp = theta_fp.cpu().detach().numpy()
        
    model = Model(n_rec, n_in)
    model.to(gpu_device)
    # define the multi target logistic regression loss function
    if solver_type == 'logistic':
        criterion = nn.BCELoss()
    else:
        criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2)
    if initialize_fp:
        model.linear.weight.data = torch.tensor(theta_fp.T, dtype=torch.float32).to(gpu_device)


    end_time = time_now.perf_counter()
    times[0] = end_time - start_time
    weights = model.linear.weight
    prd = weights[:, :n_rec].cpu().detach().numpy().flatten()
    cors[0]  = pearsonr(gnd,prd)[0];
    slopes[0]= (gnd @ prd) / (gnd@gnd)
    count_time = 0
    for idx in range(num_iters):
        start_time = time_now.perf_counter()
        d_out = model(x)

        # compute the loss
        loss = criterion((1 + d_out)/2, (1 + d)/2)
        
        # backward
        loss.backward()
        
        
        # update the weights
        optimizer.step()
         
        # zero the gradients
        optimizer.zero_grad()
        end_time = time_now.perf_counter()
        count_time= count_time + end_time- start_time;
        if np.mod(idx,1000) == 0:
            times = np.append(times,count_time);
            count_time = 0;
           
            weights = model.linear.weight
            prd = weights[:, :n_rec].cpu().detach().numpy().flatten()
            p_cor = pearsonr(gnd,prd)[0];
            cors = np.append(cors,p_cor)
            temp = time_now.localtime()
            current_time = time_now.strftime("%H:%M:%S", temp)
            slope = (gnd @ prd) / (gnd@gnd)
            slopes = np.append(slopes,slope)
            print('\t \t %s: Iteration %d finished.  Correlation %.3f. Slope %.3f. Times %.2f mins. ' \
                  %(current_time,idx + 1,p_cor,slope,np.sum(times)/60))
        if (np.sum(times)>3600*time_limit):
            break
    
    temp = time_now.localtime()
    current_time = time_now.strftime("%H:%M:%S", temp)
    print('\t %s: Solver finished in %.3f mins.' %(current_time,np.sum(times/60)))
    np.savez(save_name,times = times,slopes = slopes, cors = cors)
    
    return model.linear.weight.cpu().detach().numpy().T
 

def fit_FORCE(r,u=None,alph = 0.1,
                lam = 100,lam_in = 1,g_in = 3,verbose = 0,
                initialize_fp = 0,num_iters = 1e3, time_limit = 24,
                gnd = None,solver_type = 'currents',exp_id = 0):
    temp = time_now.localtime()
    start_time = time_now.strftime("%H:%M:%S", temp)
    print('\t %s: Running FORCE solver.' %(start_time))
    save_name = 'FORCE_exp_%d' %(exp_id);

    times  = np.zeros(1)
    slopes = np.zeros(1)
    cors   = np.zeros(1)

    start_time = time_now.perf_counter()
    
    
    def predict_single(r_before,u_in = None):
        if u_in is not None:
            x = np.dot(w_rec,r_before) + np.dot(w_in,u_in);
        else:
            x = np.dot(w_rec,r_before);
        rout = (1-alph) * r_before + alph * np.tanh(x)
        
        return rout,x
    
    if initialize_fp:
        if u is not None:
            n_in = u.shape[1]
        else:
            n_in = 0
        r_in = r[:-1,:]
        r_out = r[1:,:]
        d = (r_out - (1-alph)*r_in)/alph
        d[d<=-1+1e-6] = -1 + 1e-6;
        d[d>=1-1e-6]  = 1-1e-6;
        if n_in >0:
            x = np.c_[r_in,u];
        else:
            x = r_in.copy()
        T_data = x.shape[0]
        A = x.T @ x + lam/T_data * np.diag(np.ones(x.shape[1]));
        
        
        Ainv = inv(A);
        Xp   = Ainv @ x.T;
        z    = np.arctanh(d);
        theta_fp = Xp @ z;

    
    T = r.shape[0]-1
    n_rec=  r.shape[1]
    if u is not None and u.shape[0] != T:
        raise Exception('There is a missmatch between u and r dimensions!!!')
    
   
    
    temp=np.random.normal(0,g_in/np.sqrt(n_rec),n_rec*n_rec).reshape(n_rec,n_rec)
    np.fill_diagonal(temp,0)
    w_rec = temp
    
    if u is not None:
        n_in = u.shape[1]
        w_in =np.random.normal(0,g_in/np.sqrt(n_in),n_in*n_rec).reshape(n_rec,n_in)
    else:
        n_in = 0
    
    if initialize_fp:
        w_rec = theta_fp[:n_rec,:].T
        if n_in >0:
            w_in = theta_fp[n_rec:,:].T.reshape(n_rec,n_in)

    end_time = time_now.perf_counter()
    times[0] = end_time - start_time
    prd = w_rec.flatten()
    cors[0]  = pearsonr(gnd,prd)[0];
    slopes[0]= (gnd @ prd) / (gnd@gnd)
    

    for k in range(num_iters):
        P = np.diag(np.ones(n_rec))/lam;
        if n_in > 0:
            P_in = np.diag(np.ones(n_in))/lam_in;
        r_bef = r[0,:];
        for i in range(T):
            if np.mod(i,1000) == 0:
                start_time = time_now.perf_counter()
            temp = (r[i+1,:] - (1-alph)*r[i,:])/alph
            temp[temp>=1-1e-6] = 1-1e-6
            temp[temp<=-1+1e-6] = -1+1e-6
            
            x_now = np.arctanh(temp)
            if n_in > 0:
                r_out,x_out = predict_single(r_bef,u[i])
            else:
                r_out,x_out = predict_single(r_bef)
            
            if solver_type == 'currents':
                e_min = x_out - x_now;
            elif solver_type == 'firing_rates':
                e_min = (r_out - r[i+1,:])/alph;
                
            Pxr   = P @ r_bef;
            rxPxr = r_bef @ (P @ r_bef) + 1
            delP  = - np.outer(Pxr, Pxr) / rxPxr
            P     = P + delP;
            delW  = - np.outer(e_min,  P @ r_bef)
            
            
            w_rec = w_rec + delW;
            
            if n_in > 0:
                Pxu = P_in @ u[i]
                uxPxu = u[i] @ (P_in @ u[i]) + 1
                delP  = - np.outer(Pxu, Pxu) / uxPxu
                P_in     = P_in + delP;
                delW  = - np.outer(e_min,  P_in @ u[i])
                w_in = w_in + delW;
                r_bef,x_pred = predict_single(r_bef,u[i])
            else:
                r_bef,x_pred = predict_single(r_bef)
            
            if solver_type == 'currents':
                e_pls = x_pred - x_now
            elif solver_type == 'firing_rates':
                e_pls = (r_bef - r[i+1,:])/alph;

            if np.mod(i,1000) == 999:
                end_time = time_now.perf_counter()
                times = np.append(times,end_time - start_time);
               
                prd = w_rec.flatten()
                slope = (gnd @ prd) / (gnd@gnd)
                slopes = np.append(slopes,slope)
                p_cor = pearsonr(gnd,prd)[0];
                cors = np.append(cors, p_cor)
                temp = time_now.localtime()
                current_time = time_now.strftime("%H:%M:%S", temp)
                print('\t %s: Iteration %.2f finished. Correlation %.3f. Slope %.3f. Time %.2f mins' \
                  %(current_time,(k ) + i/T,p_cor,slope,np.sum(times)/60))
            if (np.sum(times)>3600*time_limit):
                temp = time_now.localtime()
                current_time = time_now.strftime("%H:%M:%S", temp)
                print('\t %s: Solver finished in %.3f mins.' %(current_time,np.sum(times/60)))
                np.savez(save_name,times = times,slopes = slopes, cors = cors)
                return w_rec

    temp = time_now.localtime()
    current_time = time_now.strftime("%H:%M:%S", temp)
    print('\t %s: Solver finished in %.3f mins.' %(current_time,np.sum(times/60)))
    np.savez(save_name,times = times,slopes = slopes, cors = cors)
    
    return w_rec

 