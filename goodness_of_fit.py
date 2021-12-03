import warnings
import sys
import os

if not sys.warnoptions:
    warnings.simplefilter("ignore")

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
from matplotlib.gridspec import GridSpec
from matplotlib import rc
rc('text', usetex=True)
import plot_cluster_utils as pcu
from matplotlib.lines import Line2D
from scipy.stats import gamma
from scipy.stats import chisquare
from scipy.stats import shapiro


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
   # Ns = count_reads.sum()
   # ns_i = count_reads.loc[species]
    return(np.mean((ns_i*(ns_i - 1)/(Ns*(Ns-1)))) - np.mean(ns_i/Ns)**2)

@jit
def SLM(x_0,tau,sgm,K,T,nse,delta_t):
     
    X = [x_0]
    sqrt_delta_t = np.sqrt(delta_t)
    sqrt_sig_tau = np.sqrt(sgm/tau)
    
    for i in range(T):
        
        X_t = X[-1]
        
        X_t += (X_t/tau)*(1 - X_t/K)*delta_t + sqrt_sig_tau*X_t*nse[i]*sqrt_delta_t        
        
        X.append(X_t)    
    
    return(X)

rel_ab = pd.read_csv(f"{config.data_directory}species/relative_abundance.txt.bz2",sep="\t",index_col=0)
host = sys.argv[2]

psamps = config.Poyet_samples[host]
sorted_species = rel_ab[psamps].T.mean().sort_values(ascending=False)
sorted_species = list(sorted_species[sorted_species > 5*1e-3].index)

species_num = int(sys.argv[1])
species = sorted_species[species_num]

anal_dir = config.analysis_directory
os.makedirs(f"{anal_dir}goodness_of_fit/{host}/{species}", exist_ok=True)

strain_total_freqs = pcu.get_strain_total_freqs(species,host)

for strain in strain_total_freqs.columns[:-2]:

    count_reads = pd.read_csv(f"{config.data_directory}species/count_reads.txt.bz2",sep="\t",index_col=0)
    Ns = count_reads.sum()
    ns_i = strain_total_freqs[strain]*Ns
    times = np.cumsum(strain_total_freqs["Date_Diffs"])
    times = list(times)
    if 0 in strain_total_freqs[strain].values:
        dropind = np.argwhere((strain_total_freqs[strain] == 0).values)[0][0]
        del times[dropind]
        strain_total_freqs = strain_total_freqs.loc[strain_total_freqs[strain] != 0]
    
    times = np.array(times)
    N = len(times)
    T = times[-1]
    count_reads = pd.read_csv(f"{config.data_directory}species/count_reads.txt.bz2",sep="\t",index_col=0)
    Ns = count_reads.sum()
    ns_i = strain_total_freqs[strain]*Ns

    strain_freq = strain_total_freqs[strain]

    beta = strain_freq.mean()**2/sigma_2_xi(Ns,ns_i)
    sigma = 2/(beta+1)
    xbar = np.mean(strain_freq)
    K = xbar/(1-sigma/2)
    afd = afd_gamma(xbar,beta)
    x0 = afd.rvs(int(1e4))
    ci = [afd.ppf(.05),afd.ppf(.95)]
    xrange = np.logspace(-10,0,int(1e6))
    xrange[0] = 0
    stationary = afd.pdf(xrange)
    
    fig, ax = plt.subplots(figsize=(6,4))  
    ax.plot(xrange,stationary)
    ax.fill_between(xrange,stationary,alpha=.1,color="firebrick",zorder=0)
    ax.hist(strain_freq,bins=np.logspace(np.log10(min(strain_freq)),np.log10(max(strain_freq)),8),density=True,alpha=.3,ec="k")
    ax.semilogx()
    ax.set_xlim([min(strain_freq)/2,1]);
    fig.savefig(f"{anal_dir}goodness_of_fit/{host}/{species}/{species}_{strain}_afd.png")

    intervals = np.diff(times)

    delta_t = 1/1000
    X_list = []
    tau = 1

    M = int(N/5)
    all_next = []

    p = []
    for _ in range(5000):
        all_next = []

        for j in range(len(intervals)):
        
            X_list = []
            for i in range(M):
                T = intervals[j]
                ns = np.random.normal(0, 1, int(T/delta_t))
                simvals = SLM(strain_freq[j],tau,sigma,K,int(T/delta_t),ns,delta_t)

                next_val = simvals[-1]
                X_list.append(next_val)
            all_next.append(X_list)
        
        all_next = np.array(all_next)
        I = 1*(pd.DataFrame(all_next).T < strain_freq.values[1:]).sum()
        I += 1
        Omq = np.unique(I,return_counts=True)
        count_dict = {}
        unique = range(1,M+2)
        for i in unique:
            count_dict[i] = np.count_nonzero(I == i)
        
        Q = sum([((count_dict[i] - (N-1)/(M+1))**2)/((N-1)/(M+1)) for i in unique])
        pval = 1 - stats.chi2.cdf(Q,df = M-2)
        p.append(pval)
        
    pval_med = np.median(p)
    kstest = stats.kstest(p,"uniform")
    
    q = stats.norm.ppf(p)
    shpro = shapiro(q)
    nrml_test = stats.normaltest(q)
    
    out_dic = {"pval_med":pval_med, "kstest_pval":kstest,"pvals":p,
               "shapiro":shpro,"normal_test":nrml_test}
    
    figh, axh = plt.subplots(figsize=(6,4))  
    axh.hist(p,density=True)
    figh.savefig(f"{anal_dir}goodness_of_fit/{host}/{species}/{species}_{strain}_pvalues.png")
    
    with open(f"{anal_dir}goodness_of_fit/{host}/{species}/{species}_{strain}_out_dic.pkl","wb") as f:
        pickle.dump(out_dic,f,protocol=2)