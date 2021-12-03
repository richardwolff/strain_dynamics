import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats, special
import sys
import config
plt.style.use('ggplot')
import os
import time
import datetime
from matplotlib import rc
import plot_cluster_utils as pcu
import figure_utils 
rc('text', usetex=True)
SMALL_SIZE=15
MEDIUM_SIZE=25
rc('legend', fontsize=SMALL_SIZE)
rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
rc('ytick', labelsize=MEDIUM_SIZE)

import numpy as np
from numba import jit
from scipy import stats
import time
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from numpy import trapz

plt.style.use("ggplot")

@jit
def SLM(x_0,tau,sgm,K,N,nse,delta_t):
     
    X = [x_0]
    sqrt_delta_t = np.sqrt(delta_t)
    sqrt_sig_tau = np.sqrt(sgm/tau)
    
    for i in range(int(1/delta_t)*N):
        
        X_t = X[-1]
        
        X_t += (X_t/tau)*(1 - X_t/K)*delta_t + sqrt_sig_tau*X_t*nse[i]*sqrt_delta_t        
        
        X.append(X_t)    
    
    return(X)

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
    for diff in date_diff_set:
        #YT_temp = YT_dic[num]
        #num = diff + num
        YT_dic[int(diff)] = np.linalg.matrix_power(Y,int(diff))
    return YT_dic


def make_markov_matrix(sigma,tau,K,xrange,delta_t):
  
    xrange_diffs = xrange[1:] - xrange[:-1]
    xrange_diffs = np.insert(xrange_diffs,0,0)
    
    Y = []
    
    for xt in xrange:
        y = conditional_prob(xt,sigma,tau,K,xrange,delta_t)
        y = np.array(y)
        y = y/np.sqrt(2*np.pi)
    
        Y.append(y)

    Y = np.array(Y)
     
    for i in range(Y.shape[0]):
    
        #Y[i] = np.where(Y[i] == 0,sys.float_info.min,Y[i])
        Y[i] = Y[i]*xrange_diffs
        Y[i] = Y[i,:]/sum(Y[i,:])
    
    Y = np.linalg.matrix_power(Y,int(1/delta_t))
    
    return(Y)

def bin_values(timeseries,xrange):

    binned_vals = []
    
    for elem in timeseries:
        diffsarr = np.abs(elem - xrange) 
        binned_vals.append(max(np.argmin(diffsarr),1))
    
    binned_vals = np.array(binned_vals)
    
    return(binned_vals)

def sigma_2_xi(Ns,ns_i):

    return(np.mean((ns_i*(ns_i - 1)/(Ns*(Ns-1)))) - np.mean(ns_i/Ns)**2)

def run_SLM(species,host,permuted = False):
    
    import warnings

    if not sys.warnoptions:
        warnings.simplefilter("ignore")
        
    colors = {1:"red",2:"blue",3:"yellow"}
    
    num_sims = int(1e4)
    
    anal_dir = config.analysis_directory

    out_dic = {}

    sys.stderr.write(f"Processing {species} {host} \n")

    os.makedirs(f"{config.analysis_directory}SLM_trajectory_figures/{host}/{species}", exist_ok=True)

    strain_total_freqs = pcu.get_strain_total_freqs(species,host)

    times = np.cumsum(strain_total_freqs["Date_Diffs"])
    times = np.array(list(times))
    N = times[-1]

    date_diffs = list(strain_total_freqs["Date_Diffs"])

    dd_set = np.array(sorted(list(set(strain_total_freqs["Date_Diffs"])))[1:])

    delta_t = 1/1000
    
    count_reads = pd.read_csv(f"{config.data_directory}species/count_reads.txt.bz2",sep="\t",index_col=0)
    Ns = count_reads.sum()
    
    #strain_total_freqs.index = strain_total_freqs["Order"]
    tau_list = np.linspace(.05,5,15)

    for strain in strain_total_freqs.columns[:-2]:  
        
        sys.stderr.write(f"Processing strain {strain}")
        
        ns_i = strain_total_freqs[strain]*Ns
        X_list = []
        X_mins = []
        tau = 1
        
        beta = (strain_total_freqs[strain].mean()**2)/sigma_2_xi(Ns,ns_i)
  #      beta = (strain_total_freqs[strain].mean()/strain_total_freqs[strain].std())**2
        sigma = 2/(beta+1)
        
        xbar = np.mean(strain_total_freqs[strain])
        K = xbar/(1-sigma/2)
        
        for i in range(int(1e3)):
        
            normdraws = np.random.normal(0, 1, int(N/delta_t))
            simvals = SLM(strain_total_freqs[strain][0],tau,sigma,K,N,normdraws,delta_t)
            simvals = np.array(simvals[::int(1/delta_t)])
            simvals = simvals[times]
            X_list.append(simvals)
            X_mins.append(min(simvals))
    
        minval = min(X_mins)/10

        xrange = np.logspace(np.log10(minval),np.log10(1),1500)
        xrange_diff = xrange[1:] - xrange[:-1]
        xrange_diff = np.insert(xrange_diff,0,0)
        
        strain_bin_list = []

        for elem in strain_total_freqs[strain]:
            
            diff_arr = np.abs(elem - xrange) 
            strain_bin_list.append(max(np.argmin(diff_arr),1))
    
        strain_bin_list = np.array(strain_bin_list)
        
        #Y = make_markov_matrix(sigma,tau2,K,xrange,delta_t)
        
        ## maximum likelihood estimate of tau
        tau_ll_list = []
        
        for tau2 in tau_list:
    
            Y = make_markov_matrix(sigma,tau2,K,xrange,delta_t)
        
            YT_l = matrix_power_table(dd_set,Y)
    
            lklhds = []

            for i in strain_total_freqs["Order"][:-1]:
        
                date_diff = strain_total_freqs.iloc[i+1]["Date_Diffs"]
                Yt = YT_l[date_diff]
                lklhds.append(Yt[strain_bin_list[i],strain_bin_list[i+1]])
        
            tau_ll_list.append(sum(np.log(lklhds)))
    
        tau = tau_list[np.argwhere(tau_ll_list == max(tau_ll_list))][0][0]
        
        #fig, ax = plt.subplots(figsize=(16,8))
        #ax.plot(tau_list,tau_ll_list)
        #ax.set_xlabel(r"$\tau$",size=25)
        #ax.set_ylabel("Log-likelihood",size=25)
        #fig.savefig(f"{anal_dir}SLM_trajectory_figures/{host}/{species}/{species}_{strain}_tau.png")
        
        #sys.stderr.write(f"tau: {tau} \n")
        
        if permuted:
            strain_total_freqs[strain] = np.random.permutation(strain_total_freqs[strain].values)
        
        
       # tau = 1
        
        Ystrain =  make_markov_matrix(sigma,tau,K,xrange,delta_t)
        YT_l = matrix_power_table(dd_set,Ystrain)
    
        ## find stationary distribution
        Q = YT_l[1]
        evals, evecs = np.linalg.eig(Q.T)
        #evec1 = evecs[:,np.isclose(evals, 1)]
        evec1 = evecs[:,np.argmin(np.abs(evals - 1))]
        #evec1 = evec1[:,0]
        stationary = evec1 / evec1.sum()
        stationary = stationary.real
        stationary[stationary <= 0] = 0
        
        fig, ax = plt.subplots(figsize=(16,8))
        ax.plot(xrange,stationary)
        ax.set_ylabel("Stationary probability",size=25)
        ax.set_xlabel("Relative abundance",size=25)
        ax.semilogx()
        fig.savefig(f"{anal_dir}SLM_trajectory_figures/{host}/{species}/{species}_{strain}_stationary.png")
        
        sys.stderr.write(f"stationary plotted \n")
        
        ## find log-likelihood/CI of data, with first point being drawn from stationary distribution
        lklhds_data = [stationary[strain_bin_list[0]]]
        
        
        perc = .1/2
        CI = {}  
        lower = np.argwhere(np.abs(np.cumsum(stationary) - perc) == min(np.abs(np.cumsum(stationary) - perc)))[0][0]
        upper = np.argwhere(np.abs(np.cumsum(stationary) - (1-perc)) == min(np.abs(np.cumsum(stationary) - (1-perc))))[0][0]
        CI[0] = (xrange[lower],xrange[upper])
        
        for i in strain_total_freqs["Order"][:-1]:
            
            date_diff = date_diffs[i+1]
            Yt = YT_l[date_diff]
            lklhds_data.append(Yt[strain_bin_list[i],strain_bin_list[i+1]])
            
            lower = np.argwhere(np.abs(np.cumsum(Yt[strain_bin_list[i],:]) - perc) == min(np.abs(np.cumsum(Yt[strain_bin_list[i],:]) - perc)))[0][0]
            upper = np.argwhere(np.abs(np.cumsum(Yt[strain_bin_list[i],:]) - (1-perc)) == min(np.abs(np.cumsum(Yt[strain_bin_list[i],:]) - (1-perc))))[0][0]
            CI[i + 1] = (xrange[lower],xrange[upper])
             
        data_ll = sum(np.log(lklhds_data))
        
        CI_vals = np.array(list(CI.values()))
        CI_df = pd.DataFrame(CI,index=["lower","upper"]).T
        CI_df["strain"] = strain_total_freqs[strain]        
        
#        good_fits = CI_df.loc[np.logical_and(CI_df["strain"] > CI_df["lower"], CI_df["strain"] < CI_df["upper"])].index

 #       good_fits = np.array(list(good_fits))
    
        ## plot confidence interval figure
#        fig, ax = plt.subplots(figsize=(16,8))
#        ax.fill_between(times,CI_vals[:,0], CI_vals[:,1], color="#2774AE", alpha=.4,label=f"{int(100*(1-2*perc))}\% CI",zorder=1)

#        ax.plot(times,strain_total_freqs[strain], 'o-',color="k",label="Outside CI",markeredgecolor='k',markerfacecolor="red",zorder=2)
 #       ax.plot(times[good_fits],np.array(strain_total_freqs[strain])[good_fits], 'o',color="k",
 #       label="Within CI",markeredgecolor='k',markerfacecolor="#FFD100",zorder=2)

 #       ax.semilogy()
 #       ax.set_xlabel("Timepoint",size=25)
 #       ax.set_ylabel("Frequency",size=25)

 #       ax.set_ylim([1e-5,1e0])

 #       fig.legend()   
 #       fig.savefig(f"{anal_dir}SLM_trajectory_figures/{host}/{species}/{species}_{strain}_CI_trajectory.png")
        
        
        ## now, simulate trajectories using Euler-Mayurama technique
        x0 = np.random.choice(xrange,num_sims,p=stationary)
        
        
        simval_list = []
        
        for i in range(num_sims):
    
            normdraws = np.random.normal(0, 1, int(N/delta_t))
            simvals = SLM(x0[i],tau,sigma,K,N,normdraws,delta_t)
            simvals = np.array(simvals[::int(1/delta_t)])
            simvals = simvals[times]
            simval_list.append(simvals)

        ## plot simulations
        fig_sim,ax_sim = plt.subplots(figsize=(16,8))

        for i in range(num_sims):
    
            ax_sim.plot(times,simval_list[i],alpha=.1,color="red",zorder=1,label=None)

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
           
                      
        binned_list = []
        for elem in simval_list:
            binned_list.append(bin_values(elem,xrange))
            
        log_liks = []

        for j in range(int(1e4)):
    
            lklhds = [stationary[binned_list[j][0]]]
    
            for i in range(1, len(times) - 1):
                dd = date_diffs[i+1]
                Yt = YT_l[dd]
    
                lklhds.append(Yt[binned_list[j][i],binned_list[j][i+1]])
        
            log_liks.append(sum(np.log(lklhds)))

        fig_ll, ax_ll = plt.subplots(figsize=(6,4))
        ax_ll.hist(log_liks,density=True,bins=30,label="Simulations")

        ax_ll.axvline(data_ll,color="k",linestyle="--",linewidth=3,label="Observed");
        fig_ll.legend()
        fig_ll.savefig(f"{anal_dir}SLM_trajectory_figures/{host}/{species}/{species}_{strain}_loglik.png")            
            
        pvalue = sum(1*(data_ll > log_liks)/len(log_liks))
                
        out_dic[strain] = {"pvalue":pvalue,"tau":tau,"sigma":sigma,"K":K,
                          "stationary":stationary,"xrange":xrange,
                           "CI":CI_df,"perc":perc,"log_liks_sim":log_liks,"log_lik_data":data_ll}
    
    
    
    fig,ax = plt.subplots(figsize=(12,8))
    for strain in strain_total_freqs.columns[:-2]:
        ax.plot(times,strain_total_freqs[strain], 'o-',color=colors[strain],label=f"Strain {strain}",markeredgecolor='k')
    
    fig.legend()
    fig.savefig(f"{anal_dir}SLM_trajectory_figures/{host}/{species}/{species}_{strain}_all_strains.png")
    
    return(out_dic)