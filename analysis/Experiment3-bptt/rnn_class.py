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

import time as time_now

class RNN():
    # Define some global variables that are mostly shared across all mice
    
    

    def __init__(self,opts=None):
        self.opts = self.get_defaults(opts);
        self.initialize_network()
        
        
    def get_defaults(self,opts = None):
        if opts is None:
            opts = {};
        
        if 'n_rec' not in opts:
            opts['n_rec'] = 30

        if 'n_in' not in opts:
            opts['n_in'] = 0
        
        if 'g' not in opts:
            opts['g'] = 3
            
        if 'sigma_input' not in opts:
            opts['sigma_input'] = 0
            
        if 'sigma_conversion' not in opts:
            opts['sigma_conversion'] = 0
            
        if 'alpha' not in opts:
            opts['alpha'] = 0.1
        
        if 'input_noise_type' not in opts:
            opts['input_noise_type'] = 'Gaussian'
            
        if 'conversion_noise_type' not in opts:
            opts['conversion_noise_type'] = 'Gaussian'
        
        if 'lambda_reg' not in opts:
            opts['lambda_reg'] = 0
            
        if 'FORCE_lam' not in opts:
            opts['FORCE_lam'] = 10
            
        if 'FORCE_lam_in' not in opts:
            opts['FORCE_lam_in'] = 0.5
            
        if 'FORCE_epoch' not in opts:
            opts['FORCE_epoch'] = 1000
            
        if 'verbose' not in opts:
            opts['verbose'] = False
        if 'parallel' not in opts:
            opts['parallel'] = 1
        if 'num_cores' not in opts:
            opts['num_cores'] = None
        if 'num_parallels' not in opts:
            opts['num_parallels'] = None
        if 'abs_error_tol' not in opts:
            opts['abs_error_tol'] = 1e-8
            
        if 'rel_error_tol' not in opts:
            opts['rel_error_tol'] = 1e-8
            
        if 'gnd' not in opts:
            opts['gnd'] = None
        if 'print_every' not in opts:
            opts['print_every'] = None
        if 'BT_stuck_handle' not in opts:
            opts['BT_stuck_handle'] = "continue"
            
        return opts
        
    def initialize_network(self):
        n_rec = self.opts['n_rec']
        n_in  = self.opts['n_in']
        g     = self.opts['g']
        
        # Initialize the input weights
        nn={}
        if self.opts['n_in']>0:
            nn['w_in']=np.random.normal(0,g/np.sqrt(n_in),n_in*n_rec).reshape(n_rec,n_in)
            
        # Initialize the recurrent weights
        temp=np.random.normal(0,g/np.sqrt(n_rec),n_rec*n_rec).reshape(n_rec,n_rec)
        np.fill_diagonal(temp,0)
        nn['w_rec']=temp
        
        
        self.rnn = nn
        
            
    
    def get_time_evolution(self,r_in = None,u = None,T = None):
        if T is None:
            T = 1000
        if r_in is None:
            r_in = np.random.uniform(-1,1,self.opts['n_rec'])
        if u is None:
            if self.opts['n_in']>0:
                u = np.zeros([T,self.opts['n_in']])
                
        
        if self.opts['n_in']>0:
            w_in  = self.rnn['w_in']
            
        w_rec = self.rnn['w_rec']
        alpha = self.opts['alpha']
        sigma_input = self.opts['sigma_input']
        sigma_conversion = self.opts['sigma_conversion']
    
        n_rec = w_rec.shape[0]
        
        r = np.zeros([T+1,n_rec])
        
        r[0,:] = r_in
        for i in range(T):
            r_temp=r[i,:];
            
            if self.opts['input_noise_type'] == 'Gaussian':
                noise_input = np.random.normal(0,sigma_input,n_rec)
            elif self.opts['input_noise_type'] == 'Laplace':
                noise_input = np.random.laplace(0,sigma_input,n_rec)
            elif self.opts['input_noise_type'] == 'Poisson':
                noise_input = np.random.poisson(sigma_input,n_rec)
                
                
            if self.opts['n_in']>0:
                z =  np.dot(w_rec,r_temp) + np.dot(w_in,u[i,:]) \
                     + noise_input;
               
                
            else:
                z = np.dot(w_rec,r_temp)  + noise_input;
                
            
            if self.opts['conversion_noise_type'] == 'Gaussian':
                noise_conversion = np.random.normal(0,sigma_conversion,n_rec)
            elif self.opts['conversion_noise_type'] == 'Laplace':
                noise_conversion = np.random.laplace(0,sigma_conversion,n_rec)
            elif self.opts['conversion_noise_type'] == 'Poisson':
                noise_conversion = np.random.poisson(sigma_conversion,n_rec)
            
            temp_val = alpha*np.tanh(z) + (1-alpha)*r_temp + alpha*noise_conversion;
            
            # Newly added error could throw firing rates out of bounds, correct for it!
            temp_val[temp_val >= 1-1e-6] = 1-1e-6;
            temp_val[temp_val<=-1+1e-6] = -1+1e-6;
            
            r[i+1,:] = temp_val;
            
            
        
        return r
    
    def run_forward_propagation(self,r,u = None):
        """
        When inputting r, make sure you input it as r[:-1,:], as we want to compute the firing rates
        at time t+1, but r[T+1,:] is not defined, as r is defined between 0 and T, both inclusive!
        """
        T = r.shape[0]
        if u is not None and u.shape[0] != T:
            raise Exception('There is a missmatch between u and r dimensions!!!')
    
        if self.opts['n_in']>0:
            w_in  = self.rnn['w_in']
        w_rec = self.rnn['w_rec']
        alpha = self.opts['alpha']
        
        
        if self.opts['n_in']>0: 
            z = np.dot(r,w_rec.T) + np.dot(u,w_in.T) 
        else:
            z = np.dot(r,w_rec.T) 
            
        r_out = alpha*np.tanh(z) + (1-alpha)*r;
        return r_out
    
    def fit_least_squares(self,r,u=None):
        
        T = r.shape[0]-1
        if u is not None and u.shape[0] != T:
            raise Exception('There is a missmatch between u and r dimensions!!!')
        
        n_rec = self.opts['n_rec']
        n_in  = self.opts['n_in']
        alpha = self.opts['alpha']
        lambda_reg = self.opts['lambda_reg']
        
        
        mask_rec = np.ones([self.opts['n_rec'],self.opts['n_rec']],dtype=bool)
        np.fill_diagonal(mask_rec,False)
        
        # Define cvxpy variables and constraints
        w_rec = cp.Variable([n_rec,n_rec])
        if n_in>0:
            w_in  = cp.Variable([n_rec,n_in])
        
        


        constraints = [
            w_rec[mask_rec==0] == 0
            ]
        
        
        r_tar = r[1:,:];
        r_in  = r[:-1,:];
        
        f_tar =  (r_tar - (1-alpha) * r_in) / alpha ;
        f_tar[f_tar>=1] = 1-1e-5;
        f_tar[f_tar<=-1] = -1+1e-5;
        z_tar = np.arctanh(f_tar)
        
        if n_in>0:
            z_in  = r_in@ w_rec.T  + u@ w_in.T  
        else:
            z_in  = r_in @ w_rec.T 
        
        # Define the least-squares objective
        objective = cp.Minimize( cp.sum(cp.square(z_in-z_tar)) + lambda_reg * cp.sum(cp.square(w_rec) ) )
        prob = cp.Problem(objective, constraints)
        error_flag = 1;
        error_tol = self.opts['error_tol'];
        
        while error_flag == 1:
            try:
                prob.solve( solver = cp.ECOS, 
                           abstol = error_tol,
                           reltol = error_tol,
                           feastol = error_tol,
                           abstol_inacc=error_tol,
                           reltol_inacc = error_tol,
                           feastol_inacc = error_tol,
                           verbose = self.opts['verbose']
                           )
                error_flag = 0;
            except:
                error_tol = error_tol * 100;
        
        
        self.rnn['w_rec'] = w_rec.value
        if n_in>0:
            self.rnn['w_in'] = w_in.value
            
    def fit_least_squares_gen(self,r_in,r_tar,u=None):
        
        T = r_in.shape[0]
        if u is not None and u.shape[0] != T:
            raise Exception('There is a missmatch between u and r dimensions!!!')
        
        n_rec = self.opts['n_rec']
        n_in  = self.opts['n_in']
        alpha = self.opts['alpha']
        lambda_reg = self.opts['lambda_reg']
        
        
        mask_rec = np.ones([self.opts['n_rec'],self.opts['n_rec']],dtype=bool)
        np.fill_diagonal(mask_rec,False)
        
        # Define cvxpy variables and constraints
        w_rec = cp.Variable([n_rec,n_rec])
        if n_in>0:
            w_in  = self.rnn['w_in']
        
        


        constraints = [
            w_rec[mask_rec==0] == 0
            ]
        
        
        f_tar =  (r_tar - (1-alpha) * r_in) / alpha ;
        f_tar[f_tar>=1] = 1-1e-5;
        f_tar[f_tar<=-1] = -1+1e-5;
        z_tar = np.arctanh(f_tar)
        
        if n_in>0:
            z_in  = r_in@ w_rec.T  + u@ w_in.T  
        else:
            z_in  = r_in @ w_rec.T 
        
        # Define the least-squares objective
        objective = cp.Minimize( cp.sum(cp.square(z_in-z_tar)) + lambda_reg * cp.sum(cp.square(w_rec) ) )
        prob = cp.Problem(objective, constraints)
        error_flag = 1;
        error_tol = self.opts['abs_error_tol'];
        
        while error_flag == 1:
            try:
                prob.solve( solver = cp.ECOS, 
                           abstol = error_tol,
                           reltol = error_tol,
                           feastol = error_tol,
                           abstol_inacc=error_tol,
                           reltol_inacc = error_tol,
                           feastol_inacc = error_tol,
                           verbose = self.opts['verbose']
                           )
                error_flag = 0;
            except:
                error_tol = error_tol * 100;
        
        
        self.rnn['w_rec'] = w_rec.value
        if n_in>0:
            self.rnn['w_in'] = w_in.value
            
    def fit_CoRNN(self,r,u=None):
        
        T = r.shape[0]-1
        if u is not None and u.shape[0] != T:
            raise Exception('There is a missmatch between u and r dimensions!!!')
        
        n_rec = self.opts['n_rec']
        n_in  = self.opts['n_in']
        alpha = self.opts['alpha']
        lambda_reg = self.opts['lambda_reg']
        
        
        mask_rec = np.ones([self.opts['n_rec'],self.opts['n_rec']],dtype=bool)
        np.fill_diagonal(mask_rec,False)
        
        # Define cvxpy variables and constraints
        w_rec = cp.Variable([n_rec,n_rec])
        if n_in>0:
            w_in  = cp.Variable([n_rec,n_in])

        constraints = [ w_rec[mask_rec==0] == 0]
        
        
        r_tar = r[1:,:];
        r_in  = r[:-1,:];
        
        z_tar = (r_tar - (1-alpha) * r_in + alpha) / (2*alpha);
        z_tar[z_tar<0] = 0
        z_tar[z_tar>1] = 1
        
        if n_in>0:
            z_in  = 2*(r_in@ w_rec.T  + u@ w_in.T  )
        else:
            z_in  = 2*(r_in @ w_rec.T  )
        
        
        
        z_tar = z_tar.flatten();
        z_in  = cp.reshape(z_in.T,(T*n_rec));
        
        # Define the CoRNN objective
        part_1 = cp.multiply(z_tar,  (cp.log_sum_exp(cp.vstack([np.zeros(T*n_rec),-z_in]),axis=0) ));
        part_2 = cp.multiply( (1-z_tar), (z_in + cp.log_sum_exp(cp.vstack([np.zeros(T*n_rec),-z_in]),axis=0) ) );
        z_tar[z_tar==0] = 1e-6
        z_tar[z_tar==1] = 1 - 1e-6
        
        min_val = -np.sum( z_tar * np.log(z_tar) + (1-z_tar) * np.log(1-z_tar)  )  
        
        
        objective = cp.Minimize( cp.sum(part_1 + part_2) + lambda_reg * cp.sum(cp.square(w_rec) ) -min_val);
        
        prob = cp.Problem(objective, constraints);
        
        error_flag = 1;
        abs_error_tol = self.opts['abs_error_tol'];
        rel_error_tol = self.opts['rel_error_tol'];
        
        
        
        while error_flag == 1:
            try:
                prob.solve( solver = cp.ECOS, 
                           abstol = abs_error_tol,
                           reltol = rel_error_tol,
                           feastol = abs_error_tol,
                           abstol_inacc= abs_error_tol,
                           reltol_inacc = rel_error_tol,
                           feastol_inacc = abs_error_tol,
                           verbose = self.opts['verbose']
                           )
                error_flag = 0;
            except:
                abs_error_tol = abs_error_tol * 100;
                
        
        self.rnn['w_rec'] = w_rec.value
        if n_in>0:
            self.rnn['w_in'] = w_in.value
            
    def fit_CoRNN_parallel(self,r,u=None):
        
        T = r.shape[0]-1
        if u is not None and u.shape[0] != T:
            raise Exception('There is a missmatch between u and r dimensions!!!')
        
        n_rec = self.opts['n_rec']
        n_in  = self.opts['n_in']
        alpha = self.opts['alpha']
        lambda_reg = self.opts['lambda_reg']
        
        
        
        parallel_code = self.opts['parallel']
        num_parallels = self.opts['num_parallels'] 
        
        def run_single_row(i,alpha,lambda_reg,r_in,r_tar,u=None):
            if r_tar.ndim == 1:
                r_tar = np.reshape(r_tar,r_tar.shape[0],1)
            n_sub = r_tar.shape[1]
            n_rec = r_in.shape[1]
            T = r_in.shape[0]
            rec = cp.Variable([n_sub,n_rec])
            if u is not None:
                inp = cp.Variable(u.shape[1])
            else:
                inp = 0
                
            mask_rec = np.ones([n_sub,n_rec],dtype=bool)
            for k in range(len(i)):
                mask_rec[k,i[k]] = 0
            constraints = [rec[mask_rec==0]==0]
            
            if u is not None:
                z_in  = 2*(r_in@ rec.T  + u@ w_in.T  )
            else:
                z_in  = 2*(r_in @ rec.T  )
                
            z_tar = (r_tar - (1-alpha) * r_in[:,i] + alpha) / (2*alpha);
            
            z_tar[z_tar<0] = 0
            z_tar[z_tar>1] = 1 
            z_in  = cp.reshape(z_in.T,(T*n_sub));
            z_tar = z_tar.flatten()
            
            part_1 = cp.multiply(z_tar,  (cp.log_sum_exp(cp.vstack([np.zeros(T*n_sub),-z_in]),axis=0) ));
            part_2 = cp.multiply( (1-z_tar), (z_in + cp.log_sum_exp(cp.vstack([np.zeros(T*n_sub),-z_in]),axis=0) ) );
            
            z_tar[z_tar==0] = 1e-6
            z_tar[z_tar==1] = 1 - 1e-6
            
            min_val = -np.sum( z_tar * np.log(z_tar) + (1-z_tar) * np.log(1-z_tar)  )
            
            
            objective = cp.Minimize( cp.sum(part_1 + part_2)  + lambda_reg * cp.sum(cp.square(rec) )  - min_val );
            
            
            
            prob = cp.Problem(objective, constraints);

            abs_error_tol = self.opts['abs_error_tol'];
            rel_error_tol = self.opts['rel_error_tol'];
  
            
            prob.solve( solver = cp.ECOS, 
                       abstol = abs_error_tol,
                       reltol = rel_error_tol,
                       feastol = abs_error_tol,
                       abstol_inacc= abs_error_tol,
                       reltol_inacc = rel_error_tol,
                       feastol_inacc = abs_error_tol,
                       verbose = self.opts['verbose']
                       )
               
 
                    
            
            if u is not None:
                inp = inp.value
            
            return rec.value, inp
        
        
        w_rec = np.zeros([n_rec,n_rec])
        if n_in > 0 :
            w_in = np.zeros([n_rec,n_in])
        
        r_tar = r[1:,:];
        r_in  = r[:-1,:];
        
        if parallel_code == 0:
            
            if num_parallels == None:
                num_parallels = 1
            
            ind = []; 
            for a in range(num_parallels):
                ind.append([])
            
            k = 0
            for a in range(n_rec):
                ind[k].append(a)
                k = np.mod(k+1,num_parallels)
            for i in range(num_parallels):
                rec,inp = run_single_row(ind[i],alpha,lambda_reg,r_in,r_tar[:,ind[i]],u)
                w_rec[ind[i],:] = rec
                if n_in>0:
                    w_in[ind[i],:] = inp
        else:
            if self.opts['num_cores'] == None:
                num_cores = multiprocessing.cpu_count()
            else:
                num_cores = self.opts['num_cores']
                
            if num_parallels == None:
                num_parallels = num_cores
            
            ind = []; 
            for a in range(num_parallels):
                ind.append([])
            
            k = 0
            for a in range(n_rec):
                ind[k].append(a)
                k = np.mod(k+1,num_parallels)
            
            (result) = Parallel(n_jobs=num_cores)(delayed(run_single_row)(ind[i],alpha,lambda_reg,r_in,r_tar[:,ind[i]],u) for i in range(num_parallels))
            for i in range(num_parallels):
                result_temp = result[i]
                w_rec[ind[i],:] = np.array(result_temp[0])
                if n_in>0:
                    w_in[ind[i],:] = np.array(result_temp[1])
            
            
        
        
        self.rnn['w_rec'] = w_rec
        if n_in>0:
            self.rnn['w_in'] = w_in
            
    def predict_single(self,r_before,u_in = None):
        n_in  = self.opts['n_in']
        alpha = self.opts['alpha'];
        if n_in>0:
            x = np.dot(self.rnn['w_rec'],r_before) + np.dot(self.rnn['w_in'],u_in);
        else:
            x = np.dot(self.rnn['w_rec'],r_before);
        rout = (1-alpha) * r_before + alpha * np.tanh(x)
        
        return rout,x
            
    def fit_FORCE(self,r,u=None):
        T = r.shape[0]-1
        if u is not None and u.shape[0] != T:
            raise Exception('There is a missmatch between u and r dimensions!!!')
        
        if self.opts['verbose'] == True:
            print('Force solver started.')
        
        n_rec = self.opts['n_rec']
        alpha = self.opts['alpha']
        lam   = self.opts['FORCE_lam']
        epoch = self.opts['FORCE_epoch']
        n_in  = self.opts['n_in']
        if n_in>0:
            lam_in = self.opts['FORCE_lam_in']
        
        
        for k in range(epoch):
            P = np.diag(np.ones(n_rec))/lam;
            if n_in > 0:
                P_in = np.diag(np.ones(n_in))/lam_in;
            r_bef = r[0,:];
            for i in range(T):
                temp = (r[i+1,:] - (1-alpha)*r[i,:])/alpha
                temp[temp>=1] = 1-1e-7
                temp[temp<=-1] = -1+1e-7
                
                x_now = np.arctanh(temp)
                if n_in > 0:
                    r_out,x_out = self.predict_single(r_bef,u[i])
                else:
                    r_out,x_out = self.predict_single(r_bef)
                
                
                e_min = x_out - x_now;
                Pxr   = P @ r_bef;
                rxPxr = r_bef @ (P @ r_bef) + 1
                delP  = - np.outer(Pxr, Pxr) / rxPxr
                P     = P + delP;
                delW  = - np.outer(e_min,  P @ r_bef)
                np.fill_diagonal(delW,0)
                
                self.rnn['w_rec'] = self.rnn['w_rec'] + delW;
                
                if n_in > 0:
                    Pxu = P_in @ u[i]
                    uxPxu = u[i] @ (P_in @ u[i]) + 1
                    delP  = - np.outer(Pxu, Pxu) / uxPxu
                    P_in     = P_in + delP;
                    delW  = - np.outer(e_min,  P_in @ u[i])
                    self.rnn['w_in'] = self.rnn['w_in'] + delW;
                    r_bef,x_pred = self.predict_single(r_bef,u[i])
                else:
                    r_bef,x_pred = self.predict_single(r_bef)
                e_pls = x_pred - x_now
                
            if self.opts['print_every'] is None:
                self.opts['print_every'] = 50
            
                
            if (self.opts['verbose'] and np.mod(k,self.opts['print_every'])==0):
                if self.opts['gnd'] is not None:
                    prd = self.rnn['w_rec'].flatten()
                    temp = pearsonr(prd,self.opts['gnd'])[0]
                    print('Epoch %d. Accuracy %.3f' %(k,temp))
                else:
                    print('Epoch %d. Error %.3f. Convergence ratio %.3f' %(k,np.mean(abs(e_min)),
                                                        np.mean(abs(e_pls/e_min))))
                
            
        conv_ratio = e_pls/e_min;
        return conv_ratio
        
        
        
        
        
 
