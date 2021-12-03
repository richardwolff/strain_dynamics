import sys
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import config
import seaborn as sns
import SLM_utils as slm
plt.style.use("bmh")
import figure_utils
from scipy import stats 
from scipy import special
from numba import jit
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib import rc
rc('text', usetex=True)
import plot_cluster_utils as pcu
from matplotlib.lines import Line2D
from scipy.stats import gamma
import os 
import pickle

SMALL_SIZE=15
MEDIUM_SIZE=20

rc('legend', fontsize=MEDIUM_SIZE)
rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
rc('ytick', labelsize=SMALL_SIZE)  
colors = ['#348ABD', '#A60628', '#7A68A6', '#467821']

def afd_gamma(X_bar,beta):

    return gamma(beta,scale=X_bar/beta)

def sigma_2_xi(Ns,ns_i):

    return(np.mean((ns_i*(ns_i - 1)/(Ns*(Ns-1)))) - np.mean(ns_i/Ns)**2)

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

import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

out_dic = {}

rel_ab = pd.read_csv(f"{config.data_directory}species/relative_abundance.txt.bz2",sep="\t",index_col=0)
host = sys.argv[2]

psamps = config.Poyet_samples[host]
sorted_species = rel_ab[psamps].T.mean().sort_values(ascending=False)
sorted_species = list(sorted_species[sorted_species > 5*1e-3].index)

species_num = int(sys.argv[1])
species = sorted_species[species_num]

strain_total_freqs = pcu.get_strain_total_freqs(species,host)

os.makedirs(f"{config.analysis_directory}AFD_figures/{host}/{species}", exist_ok=True)

count_reads = pd.read_csv(f"{config.data_directory}species/count_reads.txt.bz2",sep="\t",index_col=0)
Ns = count_reads.sum()

anal_dir = config.analysis_directory

for strain in strain_total_freqs.columns[:-2]:    
    
    ns_i = strain_total_freqs[strain]*Ns
    times = np.cumsum(strain_total_freqs["Date_Diffs"])
    times = list(times)
    if 0 in strain_total_freqs[strain].values:
        dropind = np.argwhere((strain_total_freqs[strain] == 0).values)[0][0]
        del times[dropind]
        strain_total_freqs = strain_total_freqs.loc[strain_total_freqs[strain] != 0]
    times = np.array(times)
    N = times[-1]
    count_reads = pd.read_csv(f"{config.data_directory}species/count_reads.txt.bz2",sep="\t",index_col=0)
    Ns = count_reads.sum()
    ns_i = strain_total_freqs[strain]*Ns
    beta = strain_total_freqs[strain].mean()**2/sigma_2_xi(Ns,ns_i)
    sigma = 2/(beta+1)
    xbar = np.mean(strain_total_freqs[strain])
    K = xbar/(1-sigma/2)
    afd = afd_gamma(xbar,beta)

    x0 = afd.rvs(int(1e4))
    ci = [afd.ppf(.05),afd.ppf(.95)]

    delta_t = 1/2500
    X_list = []
    tau = 1

    for i in range(int(1e3)):
    
        ns = np.random.normal(0, 1, int(N/delta_t))
        simvals = SLM(x0[i],tau,sigma,K,N,ns,delta_t)
        simvals = np.array(simvals[::int(1/delta_t)])
        simvals = simvals[times]
        X_list.append(simvals)
 
    fig,ax = plt.subplots(figsize=(16,8))

    for i in range(int(1e2)):
    
        ax.plot(times,X_list[i],alpha=.1,color="red",zorder=1,label=None)

    ax.plot(times,strain_total_freqs[strain], 'o-',color="k",
        label="Strain frequency",markeredgecolor='k',markerfacecolor="#2774AE",zorder=2,lw=3,
        markersize=10)

    ax.axhline(K,color="k",lw = 4,label=r"Carrying capacity $K$",linestyle="--",zorder=1)

    ax.fill_between(times,ci[0],ci[1],alpha=.2,color="green",label="90\% Stationary CI",zorder=0)


    handles, labels = plt.gca().get_legend_handles_labels()

    line = Line2D([0], [0], label = 'Simulations',color='red')

    handles.extend([line])

    ax.set_ylim([.1*ci[0],1])

    ax.set_yscale("log")

    ax.set_ylabel("Frequency",size=25)

    ax.set_xlabel("Timepoint",size=25)

    fig.legend(handles=handles);

    fig.savefig(f"{anal_dir}AFD_figures/{host}/{species}/{species}_{strain}_simulations.png")

    log_liks = []

    for j in range(int(1e3)):
    
        sim = X_list[j]
    
        lklhds = afd.pdf(sim)
        log_liks.append(sum(np.log(lklhds)))
    
    dat_ll = sum(np.log(afd.pdf(strain_total_freqs[strain])))
    
    pvalue = sum(1*(dat_ll > log_liks)/len(log_liks))
    
    sys.stderr.write(f"Strain {strain} processed! \n")
   
    out_dic[strain] = {"pvalue":pvalue,"sigma":sigma,"K":K}

with open(f"{anal_dir}AFD_figures/{host}/{species}/{species}_out_dic.pkl","wb") as f:
    pickle.dump(out_dic,f,protocol=2)