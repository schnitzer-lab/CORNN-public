

#load_ext autoreload
from RNN_lib import CustomRNN, K_bit_flip_flop, coherence_task, train
import numpy as np
import torch
import torch.nn as nn
import time as time_now
from rnn_class import RNN
import matplotlib.pyplot as plt



def get_data_from_kbit_task(num_trials,alph,n_rec,num_rnn,opts = None,task_size = 100):
    
    if opts is None:
        opts = {};
    
    if 'n_rec' not in opts:
        opts['n_rec'] = n_rec
    
    if 'n_in' not in opts:
        opts['n_in'] = 3
        
    if 'verbose' not in opts:
        opts['verbose'] = 0
        
    verbose = opts['verbose']
    
    name_dataset = "kbit_data/model_kbit_%s_%s_%s.pt" %(num_rnn,alph,n_rec);
    # load the k-bit flip flop model
    kbit_rnn = torch.load(name_dataset,map_location=torch.device('cpu'))
    kbit_rnn.eval()

    # load the k-bit flip flop task
    kbit_task = torch.load("kbit_data/kbit_task.pt",map_location=torch.device('cpu'))
    kbit_task.T = task_size;
    # the get_params function returns the weights of the network, and the alpha parameter
    # in a dictionary
    kbit_rnn_params = kbit_rnn.get_params()

    # for each parameter, print the parameter name and the shape of the parameter
    # except for alpha, which is a scalar
    for param_name, param in kbit_rnn_params.items():
        if verbose == 2:
            if param_name != "alpha":
                print(param_name, param.shape)
            else:
                print(param_name, param)
            
        if param_name == "W_rec":
            w_rec = np.array(param);
        if param_name == "W_in":
            w_in = np.array(param);
            
        if param_name == "W_out":
            w_out = np.array(param);

   

    # the output is shape (batch_size, seq_len, output_dims)
    inputs, outputs = kbit_task.gen_batch(batch_size=num_trials)
    
    

    
    m1 = RNN(opts)
    m1.rnn['w_rec'] = w_rec.T.copy()
    m1.rnn['w_in'] = w_in.T.copy()
    
    kbit_out, kbit_h = kbit_rnn.run_rnn(inputs, outputs)
    
    r_all = np.zeros(kbit_h.shape)
    for k in range(kbit_h.shape[0]):
        temp = inputs[k,:,:]
        r_all[k,:,:] = m1.get_time_evolution(T = kbit_h.shape[1]-1, u =inputs[k,:,:],r_in = kbit_h[k,0,:])
        if ( (np.mod(k,100) == 99) & (verbose == 2)):
            temp = time_now.localtime()
            current_time = time_now.strftime("%H:%M:%S", temp)
            print('%s: Trial %d created' %(current_time,k))
                  

    for idx in range(num_trials):
        if idx == 0:
            r_in = r_all[idx,:-1,:];
            r_out = r_all[idx,1:,:];
            u  = inputs[idx,:,:];
        else:
            temp_in = r_all[idx,:-1,:];
            temp_out = r_all[idx,1:,:];
            r_in = np.r_[r_in,temp_in]
            r_out = np.r_[r_out,temp_out]
            temp_inp = inputs[idx,:,:];
            u = np.r_[u,temp_inp]
            
    

    rnn_gnd = {'w_rec': w_rec,
        'w_in' : w_in,
        'w_out' : w_out}

    training_vars = {'r_in': r_in,
    'r_out' : r_out,
    'u' : u}

    trial_vars = {'inputs' : inputs, 
    'outputs' : outputs}
    
    
    if verbose:
        temp = time_now.localtime()
        current_time = time_now.strftime("%H:%M:%S", temp)
        print('%s: Loaded dataset %s' %(current_time,name_dataset))



    return training_vars,rnn_gnd,trial_vars

def generate_test_data(T):
    kbit_task = torch.load("kbit_data/kbit_task.pt")
    inputs, outputs = kbit_task.gen_trial(1,T)
    return inputs,outputs
