import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
plt.style.use('ggplot')
import datetime
from matplotlib import rc
import sys
from numba import jit

def calculate_strain_abundances(species,host):
    output_directory = "/u/scratch/r/rwolff/strainfinder_input/%s" % host
    filename_prefix = "%s/%s" % (output_directory, species)
    snp_locations = pickle.load(open(filename_prefix+".strainfinder.locations.p",'rb'))
    snp_alignment = pd.read_pickle(filename_prefix+".strainfinder.p")
    snp_samples = pickle.load(open(filename_prefix+".strainfinder.samples.p",'rb'))
    snp_samples = [elem.decode("utf-8") for elem in snp_samples]

    dates = pd.read_pickle("metadata/Poyet_collection_dates.pkl")
    dates = pd.DataFrame(dates)
    dates["Collection_Date"] = pd.to_datetime(dates.Collection_Date)

    outp=pd.read_pickle(f"~/diversity_ecology/analysis/clusters/{host}/{species}_strain_frequencies.pkl")

    strain_df = pd.DataFrame(index=snp_samples,columns=outp.keys())
    for K in outp.keys():
        strain_freq_est = outp[K]["centroid"][0]
        strain_df[K] = strain_freq_est
    
    strain_df["Collection_Date"] = dates["Collection_Date"]
    strain_df["Collection_Date"] = dates["Collection_Date"]
    strain_df = strain_df.sort_values("Collection_Date")

    strain_df["Date_Diffs"] = strain_df["Collection_Date"].diff().dt.days
    strain_df["Date_Diffs"] = strain_df["Date_Diffs"].replace(0.0,.5)
    strain_df["Order"] = range(strain_df.shape[0])

    strain_freqs = pd.DataFrame(index=strain_df.index,columns=outp.keys())

    strain_freqs[list(outp.keys())] = strain_df[outp.keys()]
    out_strain_num = max(list(outp.keys())) + 1
    strain_freqs[out_strain_num] = 1 - strain_freqs.T.sum()

    rel_ab = pd.read_csv("/u/scratch/r/rwolff/merged_MIDAS_output/%s/species/relative_abundance.txt" % host,sep="\t",index_col=0)
    spec_rel_ab = rel_ab.loc[species]
    spec_rel_ab = spec_rel_ab.loc[strain_df.index]
    strain_total_freqs = (strain_freqs.T*spec_rel_ab).T
    
    return(strain_total_freqs,strain_df)

def get_freq_diffs(strain_total_freqs,strain_df,max_succ_times):
    
    freq_diffs = []
    
    for i in range(1,strain_total_freqs.shape[0]):
        
        if strain_df.iloc[i]["Date_Diffs"] < max_succ_times:
            freq_diffs.append(np.abs(strain_total_freqs.iloc[i] - strain_total_freqs.iloc[i-1]))

    freq_diffs = pd.DataFrame(freq_diffs) 
    
    return(freq_diffs)

def calculate_mean_freq_diffs(freq_diffs,strain_total_freqs):
    
    mean_ab = strain_total_freqs.mean()
    mean_diff = freq_diffs.mean()
    
    return(list(mean_ab),list(mean_diff))

def get_freq_ratios(strain_total_freqs,strain_df,max_succ_times):
    
    freq_ratios = []
    
    for i in range(1,strain_total_freqs.shape[0]):
        
        if strain_df.iloc[i]["Date_Diffs"] < max_succ_times:
            freq_ratios.append(strain_total_freqs.iloc[i]/strain_total_freqs.iloc[i-1])

    freq_ratios = pd.DataFrame(freq_ratios) 
    
    return(freq_ratios)

def calculate_mean_freq_ratios(freq_ratios,strain_total_freqs):
    
    mean_ab = strain_total_freqs.mean()
    mean_ratio = freq_ratios.mean()
    
    return(list(mean_ab),list(mean_ratio))

@jit
def simulate_SLM(x_0,tau,sgm,K,N,nse):
     
    X = [x_0]
    
    for i in range(N):
        
        X_t = X[-1]
        
        X_t += (X_t/tau)*(1 - X_t/K) + np.sqrt(sgm/tau)*X_t*nse[i]
        
        if X_t < 0:
            X_t = 1e-6
            
        elif X_t > 1:
            X_t = 1-1e-6
        
        X.append(X_t)    
    
    return(X)

