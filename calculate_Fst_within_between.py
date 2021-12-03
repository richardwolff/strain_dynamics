import numpy as np
import pandas as pd
import config
import time
import sys

cohort = "Korpela"
data_dir = f"/u/project/ngarud/Garud_lab/metagenomic_fastq_files/{cohort}/data/"
species = sys.argv[1]
#host = sys.argv[2]
snps_dir = f"{data_dir}snps/{species}"

#data_dir = config.data_directory
#snps_dir = f"{data_dir}snps/{species}"

## implement pi within estimate from Schloissnig (2013)
chunk_size = 40000

samples_host = list(pd.read_csv(f"{snps_dir}/snps_depth.txt.bz2",sep="\t",index_col=0, nrows=0))

pi_vals = pd.DataFrame(0,index=samples_host,columns=samples_host)
df_depth_reader = pd.read_csv(f"{snps_dir}/snps_depth.txt.bz2",sep="\t",index_col=0,iterator=True,low_memory=False)
df_refreq_reader = pd.read_csv(f"{snps_dir}/snps_ref_freq.txt.bz2",sep="\t",index_col=0, iterator=True,low_memory=False)

df_depth = df_depth_reader.get_chunk(1)
df_refreq = df_refreq_reader.get_chunk(1)

#tp_matched = pd.read_pickle(f"{config.metadata_directory}Poyet_collection_dates.pkl")

t1 = time.time()
reader=True
i=0
G=pd.DataFrame(0,index=samples_host,columns=samples_host)

min_depth = 1

while reader:
    
    df_depth = df_depth_reader.get_chunk(chunk_size)
    df_refreq = df_refreq_reader.get_chunk(chunk_size)
    
    df_depth = df_depth[samples_host]
    df_refreq = df_refreq[samples_host]
    
    if df_depth.shape[0] < chunk_size:
        reader=False
        print("Complete")
    
    df_refreq = df_refreq.where(df_refreq > .05,0)
    df_refreq = df_refreq.where(df_refreq < .95,1)

    df_altfreq = 1 - df_refreq
    df_altfreq = df_altfreq.where(df_depth > min_depth, 0)
    df_refreq = df_refreq.where(df_depth > min_depth, 0)
    df_pi_mat = (df_refreq.T @ df_altfreq) + (df_altfreq.T @ df_refreq)
    G += (1*(df_depth.T > min_depth)) @ (1*(df_depth > min_depth))
    
    pi_vals += df_pi_mat

    i+=1
    
    t2 = time.time()
    sys.stderr.write(f"step {i}, time elapsed {t2 - t1}. Total sites: {G.iloc[0][0]/(chunk_size*i)} \n")
    
pi = pi_vals/G
#fst = pd.DataFrame(index=pi.index,columns=pi.columns)
#for s1 in pi.columns:
#    for s2 in pi.index:
#        fst.loc[s1,s2] = 1 - ((pi.loc[s1,s1]+pi.loc[s2,s2])/2)/pi.loc[s1,s2]
        
import os
out_path = f"{config.analysis_directory}/pi/{cohort}/{species}"
os.makedirs(out_path, exist_ok=True)

pi.to_csv(f"{out_path}/{species}_pi.txt")
#fst.to_csv(f"{out_path}/{species}_fst.txt")

sys.stderr.write(f"Files written \n \n \n")