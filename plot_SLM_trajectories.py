#!/usr/bin/python3
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
import sys
import config
plt.style.use('ggplot')
import os
from matplotlib import rc
import plot_cluster_utils as pcu

rc('text', usetex=True)
SMALL_SIZE=15
MEDIUM_SIZE=20
rc('legend', fontsize=SMALL_SIZE)
rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
rc('ytick', labelsize=SMALL_SIZE)

from numba import jit
from numpy import trapz

@jit
def conditional_prob(xt,sigma,tau,K,xrange,delta_t):
    
    mu = xt + (xt/tau*(1-xt/K) * delta_t)
    std = np.sqrt(sigma/tau) * xt * np.sqrt(delta_t)
    
    cond_prob = []
    
    for xval in xrange:
        cond_prob.append((1/std)*np.exp(-1/2*((xval - mu)/std)**2))
    
    return(cond_prob)
    
#@jit
def matrix_power_table(date_diff_set,Y):
    
    YT_dic = {1:Y}
    num = 1
    for diff in date_diff_set:
        YT_temp = YT_dic[num]
        #num = diff + num
        YT_dic[diff] = np.linalg.matrix_power(YT_temp,diff)
    return YT_dic


def make_markov_matrix(sigma,tau,K,xrange,delta_t):
    xrange_diff = (xrange[1:] - xrange[:-1])
    xrange_diff = np.insert(xrange_diff,0,xrange[0])
    
    Y = [[0]*len(xrange)]
    for xt in xrange[1:]:
        y = conditional_prob(xt,sigma,tau,K,xrange,delta_t)
        y = np.array(y)
        y = y/np.sqrt(2*np.pi)
    
        Y.append(y)

    Y = np.array(Y)
     
    for i in range(Y.shape[0]):
    
        Y[i] = np.where(Y[i] == 0,sys.float_info.min,Y[i])
        Y[i] = Y[i]*xrange_diff
        Y[i] = Y[i,:]/sum(Y[i,:])
    
    Y = np.linalg.matrix_power(Y,int(1/delta_t))
    
    return(Y)


def sim_SLM_MC(YT_dic,xrange,stationary,date_diff):
    
    x0 = np.random.choice(xrange,p=stationary)
    
    diffarr = np.abs(xrange - x0) 
    X = [xrange[np.argwhere(diffarr == min(diffarr))[0][0]]]    
    N = len(date_diff)
    for i in range(N-1):
        dd = date_diff[i+1]
        Yt = YT_dic[dd]
        x = X[-1]
        diffarr = np.abs(xrange - x) 
        argw = np.argmin(diffarr)
        
        X.append(np.random.choice(xrange,p=Yt[argw]))
        
    return X

def run_SLM(species,host):
    
    anal_dir = config.analysis_directory

    out_dic = {}

    sys.stderr.write(f"Processing {species} {host} \n")

    path = f"{anal_dir}SLM_trajectory_figures/{host}/{species}"

    os.makedirs(path, exist_ok=True)

    strain_total_freqs = pcu.get_strain_total_freqs(species,host)
    strain_total_freqs.index = strain_total_freqs["Order"]
    
    delta_t = 1/96

    tau_list = np.linspace(.1,10,20)

    dd_set = np.array(sorted(list(set(strain_total_freqs["Date_Diffs"])))[1:])
    
    times = np.cumsum(strain_total_freqs["Date_Diffs"])
    
    for strain in strain_total_freqs.columns[:-2]:
        
        #num_days = 100
        #day_num = np.argmin(np.abs(list(times - num_days)))
        #day_num = int(strain_total_freqs.shape[0]/3)
        #beta = (strain_total_freqs[strain][:day_num].mean()/strain_total_freqs[strain][:day_num].std())**2
        beta = (strain_total_freqs[strain].mean()/strain_total_freqs[strain].std())**2
        sigma = (2/(beta+1))
        
        N = int(max(list(np.cumsum(strain_total_freqs["Date_Diffs"]))))
        #K = sum(strain_total_freqs[strain]*strain_total_freqs["Date_Diffs"])/N
        K = strain_total_freqs[strain].mean()
        K = K/(1-sigma/2) 

        xrange = np.logspace(np.log10(1e-8),np.log10(1),1499)
        xrange = np.insert(xrange,0,0)
        xrange_diff = (xrange[1:] - xrange[:-1])
        xrange_diff = np.insert(xrange_diff,0,xrange[0])
        bin_list = []
        for elem in strain_total_freqs[strain]:
            diffarr = np.abs(elem - xrange)
            bin_list.append(np.argmin(diffarr))
        bin_list = np.array(bin_list)
            
        ll_list = []
        for tau2 in tau_list:
            Y = [[0]*len(xrange)]
            for xt in xrange[1:]:
                y = conditional_prob(xt,sigma,tau2,K,xrange,delta_t)
                y = np.array(y)
                y = y/np.sqrt(2*np.pi)
                Y.append(y)

            Y = np.array(Y)
            for i in range(Y.shape[0]):
                Y[i] = np.where(Y[i] == 0,sys.float_info.min,Y[i])
                Y[i] = Y[i]*xrange_diff
                Y[i] = Y[i,:]/sum(Y[i,:])
            Y = np.linalg.matrix_power(Y,int(1/delta_t))
            YT_l = matrix_power_table(dd_set,Y)
            lklhds = []

            for i in strain_total_freqs["Order"][:-1]:
                dd = strain_total_freqs.iloc[i+1]["Date_Diffs"]
                Yt = YT_l[dd]
                lklhds.append(Yt[bin_list[i],bin_list[i+1]])
            ll_list.append(sum(np.log(lklhds)))

        
        tau = tau_list[np.argwhere(ll_list == max(ll_list))][0][0]
        char_timescale = 24*delta_t*tau
        print(f"tau = {tau}")

        Y = make_markov_matrix(sigma,tau,K,xrange,delta_t)
        YT_l = matrix_power_table(dd_set,Y)
        
        bin_list = []
        
        for elem in strain_total_freqs[strain]:

                diffarr = np.abs(elem - xrange) 
                bin_list.append(np.argwhere(diffarr == min(diffarr))[0][0])
                
        bin_list = np.array(bin_list)

        lklhds = []
        CI = {}
        
        ## 90% confidence interval
        perc = .1/2
        strain_total_freqs.index = strain_total_freqs["Order"]
    
        Q = YT_l[1]
        evals, evecs = np.linalg.eig(Q.T)
        evec1 = evecs[:,np.isclose(evals, 1)]

        evec1 = evec1[:,0]

        stationary = evec1 / evec1.sum()

        stationary = stationary.real
        stationary[stationary <= 0] = sys.float_info.min
    
        lklhds = [stationary[bin_list[0]]]
        CI = {}
        
        lower = np.argwhere(np.abs(np.cumsum(stationary) - perc) == min(np.abs(np.cumsum(stationary) - perc)))[0][0]
        upper = np.argwhere(np.abs(np.cumsum(stationary) - (1-perc)) == min(np.abs(np.cumsum(stationary) - (1-perc))))[0][0]
        CI[0] = (xrange[lower],xrange[upper])
    
        for i in strain_total_freqs["Order"][:-1]:

            dd = strain_total_freqs.loc[i+1]["Date_Diffs"]
            Yt = YT_l[dd]
            lklhds.append(Yt[bin_list[i],bin_list[i+1]])
            lower = np.argwhere(np.abs(np.cumsum(Yt[bin_list[i],:]) - perc) == min(np.abs(np.cumsum(Yt[bin_list[i],:]) - perc)))[0][0]
            upper = np.argwhere(np.abs(np.cumsum(Yt[bin_list[i],:]) - (1-perc)) == min(np.abs(np.cumsum(Yt[bin_list[i],:]) - (1-perc))))[0][0]
            CI[i + 1] = (xrange[lower],xrange[upper])
        
        data_ll = sum(np.log(lklhds))
        CI_vals = np.array(list(CI.values()))
        CI_df = pd.DataFrame(CI,index=["lower","upper"]).T
        CI_df["strain"] = strain_total_freqs[strain]
        stationary_CI = [CI_vals[0,0], CI_vals[0,1]]
        
        good_fits = CI_df.loc[np.logical_and(CI_df["strain"] > CI_df["lower"], CI_df["strain"] < CI_df["upper"])].index

        good_fits = np.array(list(good_fits))
    
        ## plot confidence interval figure
        fig, ax = plt.subplots(figsize=(16,8))
        ax.fill_between(times,CI_vals[:,0], CI_vals[:,1], color="#2774AE", alpha=.4,label=f"{int(100*(1-2*perc))}\% CI",zorder=1)

        ax.plot(times,strain_total_freqs[strain], 'o-',color="k",label="Outside CI",markeredgecolor='k',markerfacecolor="red",zorder=2)
        ax.plot(times[good_fits],np.array(strain_total_freqs[strain])[good_fits], 'o',color="k",
        label="Within CI",markeredgecolor='k',markerfacecolor="#FFD100",zorder=2)

        ax.semilogy()
        ax.set_xlabel("Timepoint",size=25)
        ax.set_ylabel("Frequency",size=25)

        ax.set_ylim([1e-5,1e0])

        fig.legend()   
        fig.savefig(f"{anal_dir}SLM_trajectory_figures/{host}/{species}/{species}_{strain}_CI_trajectory.png")

        X_list = []
        date_diff = np.array(list(strain_total_freqs["Date_Diffs"]))
        
        ## run 10000 replicates of MC 
        for i in range(int(1e3)):
            X_list.append(sim_SLM_MC(YT_l,xrange,stationary,date_diff))

        from matplotlib.lines import Line2D

        fig_sim,ax_sim = plt.subplots(figsize=(16,8))

        for i in range(int(1e3)):
    
            ax_sim.plot(times,X_list[i],alpha=.1,color="red",zorder=1,label=None)

        ax_sim.plot(times,strain_total_freqs[strain], 'o-',color="k",label="Strain frequency",markeredgecolor='k',markerfacecolor="#2774AE",zorder=2,lw=3,markersize=10)

        ax_sim.axhline(K,color="k",lw = 4,label=r"Carry capacity $K$",linestyle="--",zorder=1)

        handles, labels = plt.gca().get_legend_handles_labels()

        line = Line2D([0], [0], label = 'Simulations',color='red')

        handles.extend([line])

        ax_sim.set_ylim([1e-5,1])

        ax_sim.set_yscale("log")

        ax_sim.set_ylabel("Frequency",size=25)
        ax_sim.set_xlabel("Timepoint",size=25)

        fig_sim.legend(handles=handles); 
        fig_sim.savefig(f"{anal_dir}SLM_trajectory_figures/{host}/{species}/{species}_{strain}_simulations.png")

        ll_list = []
        for X_run in X_list:

            bin_list = []

            for xi in X_run:
                diffarr = np.abs(xi - xrange) 
                bin_list.append(np.argwhere(diffarr == min(diffarr))[0][0])    
            bin_list = np.array(bin_list)
            
            lklhds = []

            for i in range(len(X_run) - 1):
                dd = date_diff[i+1]
                Yt = YT_l[dd]
                lklhds.append(Yt[bin_list[i],bin_list[i+1]])

            lklhds = np.array(lklhds)
            ll_list.append(np.sum(np.log(lklhds)))

        fig_ll, ax_ll = plt.subplots(figsize=(6,4))
        ax_ll.hist(ll_list,density=True,bins=30,label="Simulations")

        ax_ll.axvline(data_ll,color="k",linestyle="--",linewidth=3,label="Observed");
        fig_ll.legend()
        fig_ll.savefig(f"{anal_dir}SLM_trajectory_figures/{host}/{species}/{species}_{strain}_loglik.png")
        
        pval_test = np.logical_and(data_ll >= np.percentile(ll_list,5),data_ll <= np.percentile(ll_list,95))
        
        out_dic[strain] = {"pval_test":pval_test,"tau":tau,"characteristic_timescale":char_timescale,"sigma":sigma,
                          "stationary":stationary,"xrange":xrange}
        
        sys.stderr.write(f"{species} strain {strain} finished \n")   
    
    with open(f"{anal_dir}SLM_trajectory_figures/{host}/{species}/{species}_out_dic.pkl","wb") as f:
        pickle.dump(out_dic,f,protocol=2)
    
    
    
    sys.stderr.write(f"{species} finished \n")   
