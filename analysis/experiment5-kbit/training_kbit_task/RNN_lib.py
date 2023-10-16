'''
This is a library containing the customRNN class
as well as a few tasks to test it on.
This is for the CoRNN project.
'''

# import libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pickle
# import tqdm for notebook
from tqdm import tqdm_notebook as tqdm
import time

class CustomRNN(nn.Module):
    '''
    This is the custom RNN class that CoRNN assumes
    It's dynamical equation is
    h(t) = (1 - alpha) * h(t-1) + alpha * tanh(W_in * x(t-1) + W_rec * h(t-1))
    '''
    def __init__(self, input_dims, hidden_dims, output_dims, alpha=0.1, device="cpu", loss_fn=nn.MSELoss()):
        super(CustomRNN, self).__init__()
        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.output_dims = output_dims
        self.alpha = alpha
        self.device = device
        self.loss_fn = loss_fn

        # initialize weights
        self.W_in = nn.Parameter(torch.randn(input_dims, hidden_dims))
        self.W_rec = nn.Parameter(torch.randn(hidden_dims, hidden_dims))
        self.W_out = nn.Parameter(torch.randn(hidden_dims, output_dims))
        nn.init.xavier_uniform_(self.W_in)
        nn.init.xavier_uniform_(self.W_rec)
        nn.init.xavier_uniform_(self.W_out)

        # set W_rec diagonal to zero
        self.W_rec.data = self.W_rec.data * (1 - torch.eye(hidden_dims))

        # move everything to device
        self.to(device)
        
    def forward(self, x, h):
        """
        The shape of x is (batch_size, seq_len, input_dims)
        The shape of h is (batch_size, hidden_dims)
        The shape of output is (batch_size, seq_len, output_dims)
        The shape of hidden_states is (seq_len, batch_size, hidden_dims)
        """
        hidden_states = [h]
        outputs = []
        for t in range(x.size(1)):
            #h = (1 - self.alpha) * h + self.alpha * torch.tanh(fe.einsum("ij,jk->ik", x[:, t, :], self.W_in) + fe.einsum("ij,jk->ik", h, self.W_rec))
            h_from_prev = (1 - self.alpha) * h
            h_rec = torch.einsum("ij,jk->ik", h, self.W_rec)
            h_in = torch.einsum("ij,jk->ik", x[:, t, :], self.W_in)
            h = h_from_prev + self.alpha * torch.tanh(h_rec + h_in)
            hidden_states.append(h)

            # Compute the output at each timestep
            output = torch.einsum("ij,jk->ik", h, self.W_out)
            outputs.append(output)
        return outputs, hidden_states

    def get_params(self):
        # return the weights of the network, as numpy arrays
        # in a dictionary
        return {"W_in": self.W_in.detach().cpu().numpy(),
                "W_rec": self.W_rec.detach().cpu().numpy(),
                "W_out": self.W_out.detach().cpu().numpy(),
                'alpha': self.alpha}

    
    def run_rnn(self, inputs, outputs, device="cpu"):
        # inputs and outputs are numpy arrays
        x = torch.from_numpy(inputs).float() # shape (batch_size, seq_len, input_dims)
        y = torch.from_numpy(outputs).float() # shape (batch_size, seq_len, output_dims)

        # move x,y,h to device
        x = x.to(device)
        y = y.to(device)

        # Forward pass
        batch_size = x.shape[0]
        h = torch.randn(batch_size, self.hidden_dims)
        h = h.to(device)
        output, h = self(x, h)
        output = torch.stack(output, dim=1)
        h = torch.stack(h, dim=1) # shape (batch_size, seq_len, hidden_dims)

        # convert output and h to numpy
        output = output.detach().cpu().numpy()
        h = h.detach().cpu().numpy()

        return output, h

class K_bit_flip_flop():
    """
    The K-bit flip-flop task.
    """
    def __init__(self, K=3, prob_flip=0.05, T=100):
        self.K = K # dimensionality of input and output
        self.prob_flip = prob_flip # probability of a bit fliping each timestep
        self.T = T # number of timesteps per trial

    def gen_input(self):
        flip_locs = np.random.choice([0,1],
                                     size=(self.K, self.T), replace=True,
                                     p=[1-self.prob_flip, self.prob_flip])
        flip_locs = np.where(flip_locs)
        possible_bits = np.random.choice([-1,1], size=(self.K, self.T))

        inputs = np.zeros_like(possible_bits)
        inputs[flip_locs[0], flip_locs[1]] = possible_bits[flip_locs[0], flip_locs[1]]

        return inputs

    def gen_1d_output(self, inputs_1d):
        state = 0
        t = []
        for i in inputs_1d:
            if i != 0:
                state = i
            t.append(state)
        return t

    def gen_output(self, inputs):
        outputs = [self.gen_1d_output(inputs[j,:]) for j in range(self.K)]
        return np.array(outputs)

    def gen_trial(self):
        i = self.gen_input()
        o = self.gen_output(i)
        return i, o

    def gen_batch(self, batch_size):
        inputs, outputs = [], []
        for b in range(batch_size):
            i,o = self.gen_trial()
            inputs.append(i)
            outputs.append(o)
        inputs, outputs = np.array(inputs), np.array(outputs)

        # inputs is shape (batch_size, K, T)

        # transpose to format for pytorch to shape (batch_size, seq_len, input_dims)
        inputs = np.transpose(inputs, (0,2,1))
        outputs = np.transpose(outputs, (0,2,1))

        return inputs, outputs

    # visualize a trial
    def visualize_trial(self, kbit_trial_in, kbit_trial_out):
        # plot each dimension seperated in the y-axis so you can tell them apart
        # make a matrix to add to the k_bit_trial_in to make the plot
        # so that each dimension is plotted in an offset y-axis
        y_offset = np.arange(kbit_trial_in.shape[1]) * 2.5
        y_offset = np.tile(y_offset, (kbit_trial_in.shape[0], 1))
        # now plot
        plt.plot(kbit_trial_in + y_offset, color="k")
        plt.title("input and output for the first trial")
        # now do the same thing for the output, keeping the same y_offset and colors as the input
        # make it a slashed line, and set the colors to be the same colors as the inputs in every dimension
        # so we should have three different colors
        plt.plot(kbit_trial_out + y_offset, linestyle = "--", color="r")

        # replace the y-axis labels with the input dimension names
        plt.yticks(np.arange(kbit_trial_in.shape[1]) * 2.5, ["dim " + str(i) for i in range(kbit_trial_in.shape[1])])
        # make a legend that has dotted red line for the output and black line for the input
        plt.legend(["input", "output"],bbox_to_anchor=(1.3, 0.8), loc="right")
        # change the legend bar for the output to be a dotted line, and red
        plt.gca().get_legend().get_lines()[1].set_linestyle("--")
        plt.gca().get_legend().get_lines()[1].set_color("r")
        # set the x-axis
        plt.xlabel("time step")

    def visualize_rnn_output(self, y, output):
        # y is the network output, shape (batch_size, seq_len, output_dims)
        # output is the correct output, shape (batch_size, seq_len, output_dims)
        # plot the true and predicted outputs with each dimension
        # offset by 3 so they are visible, and each pair the same color
        plt.plot(y[0,:,0] + 3, 'r')
        plt.plot(output[0,:,0] + 3, 'r--')
        plt.plot(y[0,:,1] + 6, 'g')
        plt.plot(output[0,:,1] + 6, 'g--')
        plt.plot(y[0,:,2] + 9, 'b')
        plt.plot(output[0,:,2] + 9, 'b--')
        # remove the ticks on the y axis and replace with the dimension labels, 'dim 0', 'dim 1', 'dim 2'
        plt.yticks([3,6,9], ['dim 0', 'dim 1', 'dim 2'])
        plt.xlabel('Timestep')
        # add a legend that says that the solid lines are the true outputs and the dashed lines are the predicted outputs
        # but just have one legend for all three dimensions, ie they should be called 'True Output', 'Network Output'
        plt.legend(['Network Output', 'Correct Output'], 
                   bbox_to_anchor=(1.4, 0.8), loc="right")
        # make the legend lines black for both labels
        plt.gca().get_legend().get_lines()[1].set_color("k")
        plt.gca().get_legend().get_lines()[0].set_color("k")



# now we want to write another class for a task that is a bit more complicated
# we have a coherence value from -1 to 1 on every trial
# we have two input and two output dimensions
# from timesteps 25 to 75, the coherence determines the probability of having a spike on each dimension
# a coherence of -1 means that a probability of spike in 100% on the second dimension
# a coherence of 1 means that a probability of spike in 100% on the first dimension
# a coherence of 0 means that the probability of spike is 50% on each dimension and so on
# from timesteps 75 to 100 the inputs are zero
# and the output should be 1 in the first output dimension if coherence <0 and 1 in the second output dimension if coherence >0
class coherence_task():
    def __init__(self):
        self.seq_len = 100

    def gen_coherence(self):
        # generate a single coherence value
        return np.random.uniform(-1,1)

    def gen_trial(self, coherence):
        # generate a single trial
        # inputs are 2d, outputs are 2d
        # coherence determines the probability of a spike on each dimension
        # from timesteps 25 to 75
        # from timesteps 75 to 100 the inputs are zero
        # and the output should be 1 in the first output dimension if coherence <0 and 1 in the second output dimension if coherence >0
        min_val = np.round(self.seq_len / 4).astype(int);
        max_val = self.seq_len - min_val;
        size_num = max_val - min_val
        # generate the inputs
        inputs = np.zeros((self.seq_len, 2))
        inputs[min_val:max_val,0] = np.random.binomial(1, 0.5 + 0.5*coherence, size=(size_num,))
        inputs[min_val:max_val,1] = np.random.binomial(1, 0.5 - 0.5*coherence, size=(size_num,))

        # generate the outputs
        outputs = np.zeros((self.seq_len, 2))
        outputs[max_val:,0] = coherence > 0
        outputs[max_val:,1] = coherence < 0

        return inputs, outputs

    def gen_batch(self, batch_size):
        # generate a batch of trials
        inputs = np.zeros((batch_size, self.seq_len, 2))
        outputs = np.zeros((batch_size, self.seq_len, 2))
        for i in range(batch_size):
            inputs[i], outputs[i] = self.gen_trial(self.gen_coherence())
        # inputs is shape (batch_size, seq_len, input_dims)
        # outputs is shape (batch_size, seq_len, output_dims)
        
        return np.array(inputs), np.array(outputs)

    def visualize_trial(self, inputs, outputs):
        plt.plot(inputs[0,:,0], 'r')
        plt.plot(2.5+outputs[0,:,0], 'r--')
        plt.plot(-inputs[0,:,1], 'g')
        plt.plot(2.5+-outputs[0,:,1], 'g--')
        plt.ylim([-1.5,4])

        # take out y axis ticks and replace with the dimension labels, 'inputs', 'outputs'
        plt.yticks([0,2.5], ['inputs', 'outputs'])
        plt.xlabel('Timestep');

    def visualize_rnn_output(self, x, y, output):
        # x is the input, shape (batch_size, seq_len, input_dims)
        # y is the correct output, shape (batch_size, seq_len, output_dims)
        # outputs is the network output, shape (batch_size, seq_len, output_dims)
        fig, axs = plt.subplots(2, 1, sharex=True)
        axs[0].plot(x[0,:,0], 'r')
        axs[1].plot(y[0,:,0], 'r--')
        axs[0].plot(x[0,:,1], 'g')
        axs[1].plot(y[0,:,1], 'g--')
        axs[0].set_title('Inputs')
        axs[1].plot(output[0,:,0], 'r')
        axs[1].plot(output[0,:,1], 'g')
        axs[1].set_title('outputs')
        # make labels
        axs[0].set_ylabel('Input')
        axs[1].set_ylabel('Output')
        axs[1].set_xlabel('Time')
        # legend
        axs[0].legend(['Input 1', 'Input 2'])
        axs[1].legend(['Output 1', 'Output 2'])
    
    def visualize_rnn_cornn(self, x, y, output,output_cornn):
        # x is the input, shape (batch_size, seq_len, input_dims)
        # y is the correct output, shape (batch_size, seq_len, output_dims)
        # outputs is the network output, shape (batch_size, seq_len, output_dims)
        fig, axs = plt.subplots(3, 1, sharex=True)
        axs[0].plot(x[0,:,0], 'r')
        axs[1].plot(y[0,:,0], 'r--')
        axs[2].plot(y[0,:,0], 'r--')
        axs[0].plot(x[0,:,1], 'g')
        axs[1].plot(y[0,:,1], 'g--')
        axs[2].plot(y[0,:,1], 'g--')
        axs[0].set_title('Inputs')
        axs[1].plot(output[0,:,0], 'r')
        axs[1].plot(output[0,:,1], 'g')
        axs[1].set_title('outputs')
        
        axs[2].plot(output_cornn[0,:,0], 'r')
        axs[2].plot(output_cornn[0,:,1], 'g')
        axs[2].set_title('outputs (CoRNN)')
        
        
        # make labels
        axs[0].set_ylabel('Input')
        axs[1].set_ylabel('Output')
        axs[1].set_ylabel('Output (CoRNN)')
        axs[2].set_xlabel('Time')
        # legend
        axs[0].legend(['Input 1', 'Input 2'])
        axs[1].legend(['Output 1', 'Output 2'])
        axs[2].legend(['Output 1', 'Output 2'])


# make a function to do the training loop, also add a scheduler
# make sure to keep track of both the learning rate and the loss on every epoch
# return the model, the loss, and the learning rate
def train(model, task, num_epochs, batch_size, device, optim, sched=None, criterion=nn.MSELoss()):

    # Train the model, use tqdm to show the progress bar
    pbar = tqdm(range(num_epochs))
    train_losses, val_losses, lrs = [], [], []
    for epoch in pbar:

        # Generate a batch of inputs and outputs
        inputs, outputs = task.gen_batch(batch_size=batch_size)
        x = torch.from_numpy(inputs).float() # shape (batch_size, seq_len, input_dims)
        y = torch.from_numpy(outputs).float() # shape (batch_size, seq_len, output_dims)

        # move x,y,h to device
        x = x.to(device)
        y = y.to(device)

        # Forward pass
        h = torch.randn(batch_size, model.hidden_dims)
        h = h.to(device)
        output, _ = model(x, h)
        output = torch.stack(output, dim=1)
        loss = criterion(output, y)

        # Backward and optimize
        optim.zero_grad()
        loss.backward()

        # make sure that the gradients for W_rec are zero on the diagonal
        model.W_rec.grad.data = model.W_rec.grad.data - torch.diag(model.W_rec.grad.data.diag())

        optim.step()
        sched.step(loss)

        learning_rate = optim.param_groups[0]['lr']
        pbar.set_postfix({'loss': loss.item(), 'lr': learning_rate})
        if (epoch+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}', time.strftime("%H:%M:%S", time.localtime()), 'lr', learning_rate)

        # save the loss and the learning rate
        train_losses.append(loss.item())
        lrs.append(learning_rate)

    return model, np.array(train_losses), np.array(lrs)

