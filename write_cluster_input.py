import sys

import pickle
from scipy.stats import chi2
from math import log
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform
from sklearn import metrics 
import config
from numpy.random import random
import pandas as pd
import time
import numpy as np
 
data_dir = config.data_directory
species = sys.argv[1]
host = sys.argv[2]
snps_dir = f"{data_dir}snps/{species}"
species_dir = f"{data_dir}species/"
rel_ab = pd.read_csv(f"{species_dir}/relative_abundance.txt.bz2",sep="\t",index_col=0)
rel_ab = rel_ab.loc[species]
tp_matched = pd.read_pickle(f"{config.metadata_directory}Poyet_collection_dates.pkl")
samples_host = config.Poyet_samples[host]
samples_all = list(pd.read_csv(f"{snps_dir}/snps_depth.txt.bz2",sep="\t",index_col=0, nrows=0))
samples_host = [sample for sample in samples_host if sample in samples_all]

## implement pi within estimate from Schloissnig (2013)
chunk_size = 40000
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
G = 0 
#G=pd.DataFrame(0,index=samples_host,columns=samples_host)

min_depth = 5

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
        print("Complete")
        
    medians = df_depth.T.median()
    good_inds = medians[medians >= 3].index
    #good_inds = df_depth[(df_depth > 5).all(1)]
    df_depth = df_depth.loc[good_inds]
    df_refreq = df_refreq.loc[good_inds]
    
    #df_depth = df_depth[(df_depth > 0).all(1)]
    #df_depth = df_depth[(1*(df_depth > 0)).T.sum() > .9*L]
    #df_depth = df_depth[df_depth.T.median() > 5]
    
   # G += (1*(df_depth.T > 5)) @ (1*(df_depth > 5))
   # G+=1*(df_depth > 5).sum()
    
    df_refreq = df_refreq.loc[df_depth.index]
    
    df_refreq = df_refreq.where(df_refreq > .05,0)
    df_refreq = df_refreq.where(df_refreq < .95,1)
    #df_refreq = df_refreq.where(df_depth > 5, 0)
    df_refcount = df_depth*df_refreq
    df_altfreq = 1 - df_refreq
  #  df_altfreq = df_altfreq.where(df_depth > min_depth, np.nan)
  #  df_refreq = df_refreq.where(df_depth > min_depth, np.nan)
    
 #   df_pi_mat = (df_refreq.T @ df_altfreq) + (df_altfreq.T @ df_refreq)
 #   G += (1*(df_depth.T > min_depth)) @ (1*(df_depth > min_depth))
   # print(df_pi_mat)
    
    #pi_vals += df_pi_mat
    
    #df_pimat = 2*df_refreq*(df_altcount/(df_depth - 1))
    #pi_vals += df_pimat.sum()   
    
    host_refreq = df_refreq
    var_sites = np.logical_and( 1*(host_refreq > 0.05).T.sum() > L/3, 1*(host_refreq < 0.95).T.sum() > L/3 )
    var_sites_df = var_sites_df.append(host_refreq.loc[var_sites])
    var_sites_depth_df = var_sites_depth_df.append(df_depth[samples_host].loc[var_sites])
    
  #  var_sites = np.abs(df_refreq[samples_host[-1]] - df_refreq[samples_host[0]]) > .5
    
  #  var_sites_df = var_sites_df.append(df_refreq.loc[var_sites])
    #print(var_sites_df)
   # S += sum(1*(var_sites))

    i+=1
    G += sum(1*var_sites)
    
    t2 = time.time()
    print(f"step {i}, time elapsed {t2 - t1}. Total sites: {len(var_sites)} \n")
    
  #  if i > 5:
  #      reader=False
  #      print("Complete")        
#D = var_sites_depth_df.replace(np.nan,0)
#D = D.values
D = var_sites_depth_df
A = (var_sites_df*var_sites_depth_df)
#A = (var_sites_df*var_sites_depth_df).replace(np.nan,0)

anal_dir = config.analysis_directory

import os
os.makedirs(f"{anal_dir}/cluster_input/{host}",exist_ok=True)

D.to_pickle(f"{anal_dir}/cluster_input/{host}/{species}_depths.p")
A.to_pickle(f"{anal_dir}/cluster_input/{host}/{species}_alleles.p")


#x=fast_cluster_snps_by_distance(A,D)
#print(x)