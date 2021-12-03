import numpy as np
import pandas as pd
import config
import time
import sys

def get_samples(species):
    data_dir = config.data_directory
    snps_dir = f"{data_dir}snps/{species}"

    hosts = ["am","ae","ao","an"]
    tp_matched = pd.read_pickle(f"{config.metadata_directory}Poyet_collection_dates.pkl")
    samples_all = list(pd.read_csv(f"{snps_dir}/snps_depth.txt.bz2",sep="\t",index_col=0, nrows=0))

    samples_between = {}

    for host in hosts: 
        samples_host = config.Poyet_samples[host]
        samples_host = tp_matched[samples_host].sort_values().index
        samples_host = [sample for sample in samples_host if sample in samples_all]    
        L = len(samples_host)
        if L > 4:
            samples_between[host] = [samples_host[0],samples_host[int(L/4)],samples_host[int(L/2)],samples_host[int(3*L/4)], samples_host[-1]]
    
    return samples_between

## implement pi within/between estimate from Schloissnig (2013)
def calculate_pi_within_between(samples_host,species,chunk_size = 40000):
    
    if samples_host[0] == samples_host[1]:
        samples_host = [samples_host[0]]
    
    data_dir = config.data_directory
    snps_dir = f"{data_dir}snps/{species}"    
    
    pi_vals = pd.DataFrame(0,index=samples_host,columns=samples_host)
    df_depth_reader = pd.read_csv(f"{snps_dir}/snps_depth.txt.bz2",sep="\t",index_col=0,iterator=True,low_memory=False)
    df_refreq_reader = pd.read_csv(f"{snps_dir}/snps_ref_freq.txt.bz2",sep="\t",index_col=0, iterator=True,low_memory=False)
    
    df_depth = df_depth_reader.get_chunk(1)
    df_refreq = df_refreq_reader.get_chunk(1)

    tp_matched = pd.read_pickle(f"{config.metadata_directory}Poyet_collection_dates.pkl")

    t1 = time.time()
    reader=True
    i=0
    S=0
    G=pd.Series(0,index=samples_host)

    L = len(samples_host)
    var_sites_df = pd.DataFrame(columns=samples_host)
    var_sites_depth_df = pd.DataFrame(columns=samples_host)

    while reader:
    
        df_depth = df_depth_reader.get_chunk(chunk_size)
        df_refreq = df_refreq_reader.get_chunk(chunk_size)
    
        df_depth = df_depth[samples_host]
        df_refreq = df_refreq[samples_host]
    
        if df_depth.shape[0] < chunk_size:
            reader=False
            sys.stderr.write("Complete")
        
        df_depth = df_depth[(df_depth > 5).all(1)]
    
        G += df_depth.shape[0]
    
        df_refreq = df_refreq.loc[df_depth.index]
    
        df_refreq = df_refreq.where(df_refreq > .05,0)
        df_refreq = df_refreq.where(df_refreq < .95,1)
    
        df_altfreq = 1 - df_refreq
    
        df_pi_mat = (df_refreq.T.values @ df_altfreq.values) + (df_altfreq.T.values @ df_refreq.values)
    
        pi_vals += df_pi_mat
    
        var_sites = np.abs(df_refreq[samples_host[-1]] - df_refreq[samples_host[0]]) > .8
    
        var_sites_df = var_sites_df.append(df_refreq.loc[var_sites])
       # i+=1
       # if i > 3:
       #     reader=False
            
    if G[0] == 0:
        sys.stderr.write(f"Warning: no shared sites between samples {samples_host[0]} and {samples_host[1]}")
        #pi = pd.DataFrame(0)
        
    else:
        pi = pi_vals/G   
        
    return(pi,G,var_sites_df)
    
def calculate_Fst(pi):   

    Fst = 1 - ((pi.iloc[0][0] + pi.iloc[1][1])/2)/pi.iloc[0][1]
    
    return(Fst)

species = sys.argv[1]
samples = get_samples(species)

all_samples = []
for s in list(samples.values()):
    all_samples.extend(s)

pi_df = pd.DataFrame(0,index = all_samples,columns = all_samples)
fst_df = pd.DataFrame(0,index = all_samples,columns = all_samples)

hosts = list(samples.keys())
for h1 in range(len(hosts)):
    host1 = hosts[h1]
    print(f"host {host1}!!")
    sys.stderr.write(f"Starting host {host1} \n")
    for h2 in range(h1, len(hosts)):
        host2 = hosts[h2]
        sys.stderr.write(f"... comparing with host {host2} \n")        
        for s1 in samples[host1]:
            
            sys.stderr.write(f"Starting sample {s1} \n")
            
            for s2 in samples[host2]:
                        
                sys.stderr.write(f"... comparing with sample {s2} \n")
            
                pi,G,var_sites_df = calculate_pi_within_between([s1,s2], species)
            
                if s1 != s2:
                    between = pi.iloc[0][1]
                    pi_df.loc[s2,s1] = between
                    pi_df.loc[s1,s2] = between
                    #print(between)
                    fst = calculate_Fst(pi)
                    fst_df.loc[s2,s1] = fst
                    fst_df.loc[s1,s2] = fst
                    #print(fst)
                else:
                    within = pi.iloc[0][0]
                    print(within)
                    pi_df.loc[s1,s1] = within
                    #print(within)
            #print(pi_df)        
            sys.stderr.write(f"Finished sample {s1} \n \n \n")
        sys.stderr.write(f"... finished comparing {host1} with host {host2} \n \n \n") 
    sys.stderr.write(f"Finished {host1} \n \n \n")

import os
out_path = f"{config.analysis_directory}/pi/{species}"
os.makedirs(out_path, exist_ok=True)

pi_df.to_csv(f"{out_path}/{species}_pi.txt")
fst_df.to_csv(f"{out_path}/{species}_fst.txt")

sys.stderr.write(f"Files written {host1} \n \n \n")