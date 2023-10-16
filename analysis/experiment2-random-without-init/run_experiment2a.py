import numpy as np
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from rnn_class import RNN
import matplotlib.pyplot as plt
from numpy.linalg import inv
from utils import solve_corrn
from utils import solve_corrn_gpu
from utils import solve_newton_descent
from utils import solve_gradient_descent
from utils import solve_pytorch
from utils import solve_pytorch_gpu
from utils import fit_FORCE
import time as time_now

opts = {};
opts['g'] = 3; 
opts['n_rec'] = 200 # 300 for short
opts['n_in'] = 1
opts['sigma_input'] = 1e-2
opts['sigma_conversion'] = 1e-4
opts['alpha'] = 0.1
opts['input_noise_type'] = 'Gaussian'
opts['conversion_noise_type'] = 'Gaussian'
opts['verbose'] = False;
opts['lambda_reg'] = 1e-5
opts['num_cores'] = 4
opts['parallel'] = 1
T_data = 3000; # 3000 for short


num_algs = 6;
num_exps = 10;
iter_list = np.array([1e-5,1e-4,1e-3,1e-2,1e-1,1,10,100])
slopes = np.zeros([num_algs,num_exps,iter_list.shape[0]])
rmse = np.zeros([num_algs,num_exps,iter_list.shape[0]])
rmedse = np.zeros([num_algs,num_exps,iter_list.shape[0]])
cor_p = np.zeros([num_algs,num_exps,iter_list.shape[0]])
cor_s = np.zeros([num_algs,num_exps,iter_list.shape[0]])
times = np.zeros([num_algs,num_exps,iter_list.shape[0]])

idx_exp = 0;
while idx_exp<num_exps:
    try:
        temp = time_now.localtime()
        current_time = time_now.strftime("%H:%M:%S", temp)
        print('%s: Running Experiment %d' %(current_time,idx_exp))
        m1 = RNN(opts)
        r = m1.get_time_evolution(T = T_data)
        gnd = m1.rnn['w_rec'].flatten()
        #%%
        for iters in range(iter_list.shape[0]):
            lam_val = iter_list[iters];
            temp = time_now.localtime()
            current_time = time_now.strftime("%H:%M:%S", temp)
            print('\t \t %s: Running iteration %d' %(current_time,iters))
                    
            #%%
            start_time = time_now.perf_counter()
            w = solve_corrn_gpu(r[:-1,:],r[1:,:],u_in = None,l2 = lam_val*1e-5,
                            verbose = 0,gnd = gnd, threshold = 0.2,
                            num_iters = 0,
                            solver_type = 'weighted')
            times[5,idx_exp,iters] = time_now.perf_counter() - start_time;
            w_rec = w[:opts['n_rec'],:].T.flatten()
            slopes[5,idx_exp,iters] = (gnd @ w_rec) / (gnd@gnd) 
            rmse[5,idx_exp,iters] = np.sqrt(np.mean( (w_rec-gnd)**2 ))
            rmedse[5,idx_exp,iters] = np.sqrt(np.median( (w_rec-gnd)**2 ))
            cor_p[5,idx_exp,iters] = pearsonr(w_rec,gnd)[0]
            cor_s[5,idx_exp,iters] = spearmanr(w_rec,gnd)[0]
                
            temp = time_now.localtime()
            current_time = time_now.strftime("%H:%M:%S", temp)
            print('\t \t \t %s: Fixed point Cor %.3f' %(current_time,cor_p[5,idx_exp,iters]))
            #%%
            start_time = time_now.perf_counter()
            w = solve_corrn_gpu(r[:-1,:],r[1:,:],u_in = None,l2 = lam_val*1e-5,
                            verbose = 0,gnd = gnd, threshold = 0.2,
                            num_iters = 100,initialize_fp = 1,
                            solver_type = 'weighted')
            times[0,idx_exp,iters] = time_now.perf_counter() - start_time;
            w_rec = w[:opts['n_rec'],:].T.flatten()
            slopes[0,idx_exp,iters] = (gnd @ w_rec) / (gnd@gnd) 
            rmse[0,idx_exp,iters] = np.sqrt(np.mean( (w_rec-gnd)**2 ))
            rmedse[0,idx_exp,iters] = np.sqrt(np.median( (w_rec-gnd)**2 ))
            cor_p[0,idx_exp,iters] = pearsonr(w_rec,gnd)[0]
            cor_s[0,idx_exp,iters] = spearmanr(w_rec,gnd)[0]
                
            temp = time_now.localtime()
            current_time = time_now.strftime("%H:%M:%S", temp)
            print('\t \t \t %s: CoRNN finished. Cor %.3f' %(current_time,cor_p[0,idx_exp,iters]))
            
            #%%
            start_time = time_now.perf_counter()
            w  = solve_pytorch_gpu(r[:-1,:],r[1:,:],u_in = None,alph = 0.1,
                                     l2 = lam_val*1e-8,verbose = 0,gnd = gnd, #1e-7 for short
                                     initialize_fp =1,num_iters = 20000,
                                     learning_rate = .01,solver_type = 'logistic')
            times[1,idx_exp,iters] = time_now.perf_counter() - start_time;
            w_rec = w[:opts['n_rec'],:].T.flatten()
            slopes[1,idx_exp,iters] = (gnd @ w_rec) / (gnd@gnd)
            rmse[1,idx_exp,iters] = np.sqrt(np.mean( (w_rec-gnd)**2 ))
            rmedse[1,idx_exp,iters] = np.sqrt(np.median( (w_rec-gnd)**2 ))
            cor_p[1,idx_exp,iters] = pearsonr(w_rec,gnd)[0]
            cor_s[1,idx_exp,iters] = spearmanr(w_rec,gnd)[0]
            
            temp = time_now.localtime()
            current_time = time_now.strftime("%H:%M:%S", temp)
            print('\t \t \t %s: PT-logistic finished. Cor %.3f' %(current_time,cor_p[1,idx_exp,iters]))
    
            #%%
            start_time = time_now.perf_counter()
            w  = solve_pytorch_gpu(r[:-1,:],r[1:,:],u_in = None,alph = 0.1,
                                     l2 = lam_val*1e-8,verbose = 0,gnd = gnd, #1e-7 for short
                                     initialize_fp =1,num_iters = 20000,
                                     learning_rate = .01,solver_type = 'l2')
            times[2,idx_exp,iters] = time_now.perf_counter() - start_time;
            w_rec = w[:opts['n_rec'],:].T.flatten()
            slopes[2,idx_exp,iters] = (gnd @ w_rec) / (gnd@gnd)
            rmse[2,idx_exp,iters] = np.sqrt(np.mean( (w_rec-gnd)**2 ))
            rmedse[2,idx_exp,iters] = np.sqrt(np.median( (w_rec-gnd)**2 ))
            cor_p[2,idx_exp,iters] = pearsonr(w_rec,gnd)[0]
            cor_s[2,idx_exp,iters] = spearmanr(w_rec,gnd)[0]
            
            temp = time_now.localtime()
            current_time = time_now.strftime("%H:%M:%S", temp)
            print('\t \t \t %s: PT-l2 finished. Cor %.3f' %(current_time,cor_p[2,idx_exp,iters]))
    
            
            #%%
            start_time = time_now.perf_counter()
            w = fit_FORCE(r,None,alph = 0.1,
                            lam = lam_val*200,g_in = 3,verbose = 0,
                            initialize_fp = 1,num_iters = 100,
                            gnd = gnd,solver_type = 'currents')
            times[3,idx_exp,iters] = time_now.perf_counter() - start_time;
            w_rec = w[:opts['n_rec'],:].T.flatten()
            slopes[3,idx_exp,iters] = (gnd @ w_rec) / (gnd@gnd)
            rmse[3,idx_exp,iters] = np.sqrt(np.mean( (w_rec-gnd)**2 ))
            rmedse[3,idx_exp,iters] = np.sqrt(np.median( (w_rec-gnd)**2 ))
            cor_p[3,idx_exp,iters] = pearsonr(w_rec,gnd)[0]
            cor_s[3,idx_exp,iters] = spearmanr(w_rec,gnd)[0]
            
            temp = time_now.localtime()
            current_time = time_now.strftime("%H:%M:%S", temp)
            print('\t \t \t %s: Force-cur finished. Cor %.3f' %(current_time,cor_p[3,idx_exp,iters]))
            
            #%%
            start_time = time_now.perf_counter()
            w = fit_FORCE(r,None,alph = 0.1,
                            lam = lam_val*200,g_in = 3,verbose = 0,
                            initialize_fp = 1,num_iters = 100,
                            gnd = gnd,solver_type = 'firing_rates')
            times[4,idx_exp,iters] = time_now.perf_counter() - start_time;
            w_rec = w[:opts['n_rec'],:].T.flatten()
            slopes[4,idx_exp,iters] = (gnd @ w_rec) / (gnd@gnd)
            rmse[4,idx_exp,iters] = np.sqrt(np.mean( (w_rec-gnd)**2 ))
            rmedse[4,idx_exp,iters] = np.sqrt(np.median( (w_rec-gnd)**2 ))
            cor_p[4,idx_exp,iters] = pearsonr(w_rec,gnd)[0]
            cor_s[4,idx_exp,iters] = spearmanr(w_rec,gnd)[0]
            
            temp = time_now.localtime()
            current_time = time_now.strftime("%H:%M:%S", temp)
            print('\t \t \t %s: Force-fr finished. Cor %.3f' %(current_time,cor_p[4,idx_exp,iters]))
        idx_exp = idx_exp +1
    except:
        print('Error occured')




np.savez('experiment2a_results.npz',slopes = slopes,
         rmse = rmse, rmedse = rmedse,cor_p = cor_p,cor_s = cor_s,
         times = times)
