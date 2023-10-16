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


from matplotlib.cm import get_cmap
cmap = get_cmap('tab10')
color_all = [cmap(0),cmap(1),cmap(2),cmap(3),cmap(4),cmap(5),cmap(6)]


save_names = [];
for exp in range(7):
    for k in range(7):
        if k==0:
            save_names = np.append(save_names,"Experiment %d/CORNN_CPU_exp_%d.npz" %(exp,exp))
        if k==1:
            save_names = np.append(save_names,"Experiment %d/CORNN_GPU_exp_%d.npz" %(exp,exp))
        if k==2:
            save_names = np.append(save_names,"Experiment %d/Pytorch_CPU_type_logistic_fpinit_0_exp_%d.npz" %(exp,exp))
        if k==3:
            save_names = np.append(save_names,"Experiment %d/Pytorch_GPU_type_logistic_fpinit_0_exp_%d.npz" %(exp,exp))
        if k==4:
            save_names = np.append(save_names,"Experiment %d/Pytorch_GPU_type_logistic_fpinit_1_exp_%d.npz" %(exp,exp))
        if k==5:
            save_names = np.append(save_names,"Experiment %d/Pytorch_GPU_type_l2_fpinit_0_exp_%d.npz" %(exp,exp))
        if k==6:
            save_names = np.append(save_names,"Experiment %d/FORCE_exp_%d.npz" %(exp,exp))

labels = ["CORNN (CPU)", "CORNN (GPU)", "GD-logistic (CPU)", 
          "GD-logistic-fixed point initialized (GPU)",
          "GD-logistic (GPU)", "Teacher forced gradient descent  (GPU)", "FORCE (Perich et al 2021)"]

ind_interest = [0,1,6]

for k in ind_interest:
    for exp in range(7):
        f = np.load(save_names[exp*7+k])
        times = np.array(f['times'])
        acc   = np.array(f['cors'])
        #if np.mod(i,7) == 6:
        #    acc = acc[0::2]
        if exp == 0:
            times_new = np.zeros([7,times.shape[0]*2]) + np.nan
            acc_new   = np.zeros([7,acc.shape[0]*2]) + np.nan
        acc_new[exp,:acc.shape[0]] = acc
        for l in range(times.shape[0]):
            times_new[exp,l] = np.sum(times[:l+1])
        
        #plt.plot(times_new[exp,:],acc_new[exp,:],color = color_all[np.mod(k,7)],label = labels[np.mod(k,7)])
    
    plt.errorbar(np.nanmedian(times_new,0),np.nanmedian(acc_new,0),np.nanstd(acc_new,0),np.nanstd(times_new,0),color = color_all[np.mod(k,7)],label = labels[np.mod(k,7)])
    
    #plt.scatter(times_new,acc_new)

plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_linewidth(2)
plt.gca().spines['bottom'].set_linewidth(2)

plt.axvline(x=60,ls = '--',color = 'red')
plt.axvline(x=3600,ls = '--',color = 'red')
plt.axvline(x=3600*24,ls = '--',color = 'red')
plt.xscale('log')
plt.ylim([0,1])
ax.xaxis.set_minor_locator(plt.NullLocator())
plt.xlabel('Training time (s)',fontsize = 20)    
plt.ylabel('Accuracy',fontsize = 20)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.4),
          fancybox=True, shadow=True)

plt.savefig("FigureN2.pdf", bbox_inches='tight')


