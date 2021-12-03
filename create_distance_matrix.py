import numpy as np
import pandas as pd
import config
import time
import sys
import pickle 

def make_distance_matrix(f,D,min_depth=5):
    S = f.shape[0]
    dist_mat_pol1 = np.zeros((S,S))
    dist_mat_pol2 = np.zeros((S,S))
    
    for i in range(S - 1):
        
        fi1, Di1 = f[i], D[i]
        
        for j in range(i+1,S):
            
            fj1,Dj1 = f[j], D[j]
                        
            denom = (fi1 + fj1)*(1 - fi1 + 1 - fj1)
            num = 2*(Di1+Dj1)*((fi1 - fj1)**2)
            
            zeros = denom == 0
            nonzeros = zeros == False
            zero_weight = 1-len(denom[zeros])/S
            
            dist_mat_pol1[(i,j)] = zero_weight*np.nanmean(num[nonzeros]/denom[nonzeros])
            fj2 = 1 - fj1
            denom2 = (fi1 + fj2)*(1 - fi1 + 1 - fj2)
            num2 = 2*(Di1+Dj1)*(fi1 - fj2)**2

            zeros2 = denom2 == 0
            nonzeros2 = zeros2 == False
            zero_weight2 = 1-len(denom2[zeros2])/S
            
            dist_mat_pol2[(i,j)] = zero_weight2*np.nanmean(num2[nonzeros2]/denom2[nonzeros2])
            
    dist_mat_pol1 = dist_mat_pol1 + dist_mat_pol1.T
    dist_mat_pol2 = dist_mat_pol2 + dist_mat_pol2.T
    
    return(dist_mat_pol1, dist_mat_pol2)

species = sys.argv[1]
host = sys.argv[2]

var_sites_depth_df = pd.read_csv(f"{config.analysis_directory}/pi/{host}/{species}_depth.txt.bz2",index_col=0).astype(float)
var_sites_df = pd.read_csv(f"{config.analysis_directory}/pi/{host}/{species}_freqs.txt.bz2",index_col=0).astype(float)

D = var_sites_depth_df.values
f = var_sites_df.values

dist_mat_pol1,dist_mat_pol2 = make_distance_matrix(f,D)

dist_mat_dic = {"pol_1":dist_mat_pol1,"pol_2":dist_mat_pol2}

with open(f'/u/scratch/r/rwolff/distance_matrix/{host}/{species}_mat.pkl', 'wb') as handle:
    pickle.dump(dist_mat_dic, handle, protocol=2)