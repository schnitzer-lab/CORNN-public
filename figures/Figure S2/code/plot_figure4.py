import numpy as np
import matplotlib.pyplot as plt


fig, ax = plt.subplots(1, 1, figsize=(7,3))

# Set line width of axes
ax.spines["top"].set_linewidth(0)
ax.spines["bottom"].set_linewidth(2)
ax.spines["right"].set_linewidth(0)
ax.spines["left"].set_linewidth(2)

ax.tick_params(axis="both", which="both", bottom=True, top=False,
               labelbottom=True, left=True, right=False,
               labelleft=True,direction='out',length=10,width=2.0,pad=8,labelsize=20)

f = np.load('experiment1a_results_fp.npz')
slopes = f['slopes']
rmse = f['rmse']
rmedse = f['rmedse']
cor_p = f['cor_p']
cor_s = f['cor_s']
times = f['times']

# 0 is CoRNN, CPU
# 1 is Pytorch, logistic, GPU
# 2 is Pytorch, logistic, CPU
# 3 is NT, weighted
# 4 is Pytorch, l2, GPU
# 5 is NT, logistic
# 6 is Force, currents
# 7 is Force, firing rates
# 8 is CoRNN, GPU



labels_all = ["CoRNN","Gradient descent - CE", \
              "Gradient descent - CE", "Newton descent - weighted CE", "Gradient descent - L2", \
                  "Newton descent - CE", "Force - currents", "Force - firing rates", \
                  "CoRNN"]


for pick in [0,3,5,2,6,7]:

    temp = np.nanmedian(cor_p[pick,:,:],0)
    temp_std = np.std(cor_p[pick,:,:],0) / np.sqrt(cor_p.shape[1])
    x = np.nanmedian(times[pick,:,:],0)
    x_std = np.std(times[pick,:,:],0) / np.sqrt(cor_p.shape[1])
    plt.errorbar(x,temp,temp_std,x_std,label = labels_all[pick], lw=2)

plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_linewidth(2)
plt.gca().spines['bottom'].set_linewidth(2)




plt.xscale('log')
ax.xaxis.set_minor_locator(plt.NullLocator())
plt.xlim([10**(-1.9),10**(1.7)])
plt.ylim(0.86,1.0)
plt.xticks([10**(-1),10**(0),10**(1)],fontsize=20)
plt.yticks(np.linspace(0.9,1.0,3),fontsize=20)
plt.xlabel('Time (s)',fontsize = 20)    
plt.ylabel('Accuracy',fontsize = 20)

plt.savefig("Figure4a.pdf", bbox_inches='tight')










f = np.load('experiment1b_results_fp.npz')
slopes = f['slopes']
rmse = f['rmse']
rmedse = f['rmedse']
cor_p = f['cor_p']
cor_s = f['cor_s']
times = f['times']

# 0 is CoRNN, CPU
# 1 is Pytorch, logistic, GPU
# 2 is Pytorch, logistic, CPU
# 3 is NT, weighted
# 4 is Pytorch, l2, GPU
# 5 is NT, logistic
# 6 is Force, currents
# 7 is Force, firing rates
# 8 is CoRNN, GPU



fig, ax = plt.subplots(1, 1, figsize=(7,3))

# Set line width of axes
ax.spines["top"].set_linewidth(0)
ax.spines["bottom"].set_linewidth(2)
ax.spines["right"].set_linewidth(0)
ax.spines["left"].set_linewidth(2)

ax.tick_params(axis="both", which="both", bottom=True, top=False,
               labelbottom=True, left=True, right=False,
               labelleft=True,direction='out',length=10,width=2.0,pad=8,labelsize=20)


for pick in [0,3,5,2,6,7]:

    temp = np.nanmedian(cor_p[pick,:,:],0)
    temp_std = np.std(cor_p[pick,:,:],0) / np.sqrt(cor_p.shape[1])
    x = np.nanmedian(times[pick,:,:],0)
    x_std = np.std(times[pick,:,:],0) / np.sqrt(cor_p.shape[1])
    plt.errorbar(x,temp,temp_std,x_std,label = labels_all[pick], lw=2)

plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_linewidth(2)
plt.gca().spines['bottom'].set_linewidth(2)

plt.xlim([10**(-1.9),10**(1.7)])
plt.ylim(0.93,0.98)
plt.xticks([10**(-1),10**(0),10**(1)],fontsize=20)
plt.yticks([0.93,0.95, 0.97],fontsize=20)
#plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.xlabel('Time (s)',fontsize = 20)    
plt.ylabel('Accuracy',fontsize = 20)
plt.xscale('log')
ax.xaxis.set_minor_locator(plt.NullLocator())


plt.legend(loc=(1.04, 0),title="CPU implementations")
plt.savefig("Figure4b.pdf", bbox_inches='tight')