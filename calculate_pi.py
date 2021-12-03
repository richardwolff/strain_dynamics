import numpy as np
import pandas as pd
import config
from ast import literal_eval
import sys

#data_dir = config.data_directory
data_dir = "/u/project/ngarud/Garud_lab/metagenomic_fastq_files/HMP1-2/data/"
species = sys.argv[1]
#host = sys.argv[2]
snps_dir = f"{data_dir}snps/{species}"
analysis_dir = config.analysis_directory
min_depth = 5
frac_sites = 1/3

## implement pi within estimate from Schloissnig (2013)
chunk_size = 40000
samples_host = list(pd.read_csv(f"{snps_dir}/snps_depth.txt.bz2",sep="\t",index_col=0, nrows=0))
pi_vals = pd.Series(0,index=samples_host)
df_depth_reader = pd.read_csv(f"{snps_dir}/snps_depth.txt.bz2",sep="\t",index_col=0, iterator=True,low_memory=False)
df_refreq_reader = pd.read_csv(f"{snps_dir}/snps_ref_freq.txt.bz2",sep="\t",index_col=0, iterator=True,low_memory=False)

df_depth = df_depth_reader.get_chunk(1)
df_refreq = df_refreq_reader.get_chunk(1)

#tp_matched = pd.read_pickle(f"{config.metadata_directory}Poyet_collection_dates.pkl")

reader=True
i=0
S=0
G=pd.Series(0,index=samples_host)

#samples_host = config.Poyet_samples[host]
#samples_all = list(pd.read_csv(f"{snps_dir}/snps_depth.txt.bz2",sep="\t",index_col=0, nrows=0))
#samples_host = [sample for sample in samples_host if sample in samples_all]
L = len(samples_host)
var_sites_df = pd.DataFrame(columns=samples_host)
var_sites_depth_df = pd.DataFrame(columns=samples_host)
i = 0
while reader:
    
    df_depth = df_depth_reader.get_chunk(chunk_size)
    df_refreq = df_refreq_reader.get_chunk(chunk_size)
    
    if df_depth.shape[0] < chunk_size:
        reader=False
        print("Complete")
        
    medians = df_depth.T.median()
    good_inds = medians[medians >= min_depth].index
    df_depth = df_depth.loc[good_inds]
    df_refreq = df_refreq.loc[good_inds]
    
    df_refreq = df_refreq*((1*(df_depth > min_depth)).replace(0,np.nan))
    df_depth = df_depth*(1*(df_depth > min_depth)).replace(0,np.nan)
    G+=1*(df_depth > min_depth).sum()
 
    df_refreq = df_refreq*((1*(df_refreq >= .05)))
    df_refreq = df_refreq*((1*(df_refreq <= .95)))
    
    df_refcount = df_depth*df_refreq
    df_altcount = df_depth*(1-df_refreq)
    
    df_pimat = 2*df_refreq*(df_altcount/(df_depth - 1))
    pi_vals += df_pimat.sum()   
    
    #host_refreq = df_refreq[samples_host]
    #var_sites = np.logical_and(host_refreq > 0,host_refreq < 1).T.sum() > L*frac_sites
    #var_sites_df = var_sites_df.append(host_refreq.loc[var_sites])
    #var_sites_depth_df = var_sites_depth_df.append(df_depth[samples_host].loc[var_sites])

    #S += sum(1*(var_sites))
    i+=1
    sys.stderr.write(f"step {i} complete \n")
#pi = (pi_vals/G).loc[tp_matched[samples_host].sort_values().index]
pi = pi_vals/G

pi.to_csv(f"{config.analysis_directory}/pi/HMP1-2/{species}_pi.txt")
#var_sites_df.to_csv(f"{config.analysis_directory}/pi/HMP1-2/{host}/{species}_freqs.txt.bz2",compression="bz2")
#var_sites_depth_df.to_csv(f"{config.analysis_directory}/pi/HMP1-2/{host}/{species}_depth.txt.bz2",compression="bz2")
