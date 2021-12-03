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
import statsmodels.formula.api as smf
import seaborn as sns
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
    num = 1
    for diff in date_diff_set:
        YT_temp = YT_dic[num]
        YT_dic[diff] = np.linalg.matrix_power(YT_temp,diff)
    return YT_dic


def make_markov_matrix(sigma,tau,K,xrange,delta_t):
    
    Y = []
    for xt in xrange:
        y = conditional_prob(xt,sigma,tau,K,xrange,delta_t)
        y = np.array(y)
        y = y/np.sqrt(2*np.pi)
    
        Y.append(y)

    Y = np.array(Y)
     
    for i in range(Y.shape[0]):
    
        Y[i] = np.where(Y[i] == 0,sys.float_info.min,Y[i])
        Y[i] = Y[i]*xrange_diff
        #Y[i] = np.where(Y[i] == 0,sys.float_info.min,Y[i]) 
        Y[i] = Y[i,:]/sum(Y[i,:])
    
    Y = np.linalg.matrix_power(Y,int(1/delta_t))
    
    return(Y)

def bin_values(timeseries,xrange):

    binned_vals = []
    for elem in timeseries:
        diffarr = np.abs(elem - xrange) 
        binned_vals.append(max(np.argmin(diffarr),1))
    
    binned_vals = np.array(binned_vals)
    return(binned_vals)

def run_SLM(species,host):

    data_dir = config.data_directory
    meta_dir = config.metadata_directory
    species_dir = f"{data_dir}species/"
    snps_dir = f"{data_dir}snps/{species}"

    dates = pd.read_pickle(f"{meta_dir}/Poyet_collection_dates.pkl")
    dates = pd.DataFrame(dates)
    dates["Collection_Date"] = pd.to_datetime(dates.Collection_Date)
    
    rel_ab = pd.read_csv(f"{species_dir}/relative_abundance.txt.bz2",sep="\t",index_col=0)
    rel_ab = rel_ab.loc[species]
    
    samples_host = config.Poyet_samples[host]
#samples_all = list(pd.read_csv(f"{snps_dir}/snps_depth.txt.bz2",sep="\t",index_col=0, nrows=0))


    samples_host = [sample for sample in samples_host if rel_ab.loc[sample] > 0]

    samples_host_dates = dates.loc[samples_host]
    samples_host_dates = samples_host_dates.sort_values("Collection_Date")


    samples_host_dates["Date_Diffs"] = samples_host_dates["Collection_Date"].diff().dt.days
    samples_host_dates["Date_Diffs"] = samples_host_dates["Date_Diffs"].replace(0.0,1)
    samples_host_dates["Date_Diffs"][0] = 0
    samples_host_dates["Date_Diffs"] = samples_host_dates["Date_Diffs"].astype(int)
    samples_host_dates["Order"] = range(samples_host_dates.shape[0])
   
    species_df = samples_host_dates
    species_df["abundance"] = rel_ab
    print(species_df)
    times = np.cumsum(species_df["Date_Diffs"])
    times = np.array(list(times))
    N = times[-1]
    
    beta = (species_df["abundance"].mean()/species_df["abundance"].std())**2
    sigma = 2/(beta+1)

    date_diffs = list(species_df["Date_Diffs"])

    K = np.mean(species_df["abundance"])
    K = K/(1-sigma/2)

    dd_set = np.array(sorted(list(set(species_df["Date_Diffs"])))[1:])

    delta_t = 1/1000
    
    tau_list = np.linspace(.3,5,60)
    bin_list = []

    xrange = np.logspace(np.log10(1e-5),np.log10(1),1000)
    xrange_diff = xrange[1:] - xrange[:-1]
    xrange_diff = np.insert(xrange_diff,0,0)
    
    for elem in species_df["abundance"]:
        diffarr = np.abs(elem - xrange) 
        bin_loc = max(np.argwhere(diffarr == min(diffarr))[0][0],1)
    
        bin_list.append(bin_loc)
    
    bin_list = np.array(bin_list)
    bin_list
    ll_list = []
    
    k = 0
    inc = True
    while inc and k < len(tau_list):
    
        Y = []
        tau2 = tau_list[k]
        for xt in xrange:
            y = conditional_prob(xt,sigma,tau2,K,xrange,delta_t)
            y = np.array(y)
            y = y/np.sqrt(2*np.pi)
    
            Y.append(y)

        Y = np.array(Y)
     
        for i in range(Y.shape[0]):
    
            Y[i] = Y[i]*xrange_diff
        
            Y[i] = Y[i,:]/sum(Y[i,:])    
        
        Y = np.linalg.matrix_power(Y,int(1/delta_t))
    
        YT_l = matrix_power_table(dd_set,Y)
    
        lklhds = []
        
        for i in species_df["Order"][:-1]:
        
            date_diff = species_df.iloc[i+1]["Date_Diffs"]
            Yt = YT_l[date_diff]
            lklhds.append(Yt[bin_list[i],bin_list[i+1]])
        
        ll_list.append(sum(np.log(lklhds)))
        print(sum(np.log(lklhds)))
    
        if k > 1:
            if ll_list[k] < ll_list[k - 1]:
                inc = False
                print("complete")
        k += 1
    tau = tau_list[np.argwhere(ll_list == max(ll_list))][0][0]
    
    Y = []
    for xt in xrange:
        y = conditional_prob(xt,sigma,tau,K,xrange,delta_t)
        y = np.array(y)
        y = y/np.sqrt(2*np.pi)
    
        Y.append(y)

    Y = np.array(Y)    
    for i in range(len(Y)):
    
        Y[i] = np.array(Y[i])*xrange_diff
        Y[i] = Y[i,:]/sum(Y[i,:])

    Y = np.linalg.matrix_power(Y,int(1/delta_t))
    YT_l = matrix_power_table(dd_set,Y)
    
    bin_list_species = []
    for elem in species_df["abundance"]:
        diffarr = np.abs(elem - xrange) 
        bin_list_species.append(max(np.argmin(diffarr),1))
    bin_list_species = np.array(bin_list_species)
    
    Q = YT_l[1]
    evals, evecs = np.linalg.eig(Q.T)
    evec1 = evecs[:,np.isclose(evals, 1)]
    evec1 = evec1[:,0]
    stationary = evec1 / evec1.sum()
    stationary = stationary.real
    stationary[stationary <= 0] = 0
    
    lklhds_data = [stationary[bin_list_species[0]]]
    for i in species_df["Order"][:-1]:
        date_diff = date_diffs[i+1]
        Yt = YT_l[date_diff]
        lklhds_data.append(Yt[bin_list[i],bin_list[i+1]])
    data_ll = sum(np.log(lklhds_data))
    
    numsims=int(1e3)
    x0 = np.random.choice(xrange,numsims,p=stationary)
    X_list = []

    for i in range(numsims):
    
        ns = np.random.normal(0, 1, int(N/delta_t))
        simvals = SLM(x0[i],tau,sigma,K,N,ns,delta_t)
        simvals = np.array(simvals[::int(1/delta_t)])
        simvals = simvals[times]
        X_list.append(simvals)
    
    binned_list = []
    for elem in X_list:
        binned_list.append(bin_values(elem,xrange))
        
    log_liks = []

    for j in range(numsims):
    
        lklhds = [stationary[binned_list[j][0]]]
    
        for i in range(1, len(times) - 1):
            dd = date_diffs[i+1]
            Yt = YT_l[dd]
        
            lklhds.append(Yt[binned_list[j][i],binned_list[j][i+1]])
        
        log_liks.append(sum(np.log(lklhds)))
    
    SLM_pval = sum(1*(data_ll > log_liks))/len(log_liks)
    
    return({"beta":beta, "sigma":sigma, "K":K, "SLM_pval": SLM_pval})