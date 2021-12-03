import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
import sys
import config
import os
from matplotlib import rc
import plot_cluster_utils as pcu
import figure_utils 
rc('text', usetex=False)
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
from sklearn.linear_model import LinearRegression
from matplotlib.lines import Line2D
import statsmodels.formula.api as smf
import species_SLM_test
import seaborn as sns
plt.style.use("ggplot")

data_dir = config.data_directory
species_dir = f"{data_dir}species/"
metadir = config.metadata_directory
dates = pd.read_pickle(f"{metadir}Poyet_host_samples_timepoints.pkl")

out_df = pd.DataFrame(columns=["pi_b_slope","pi_b_pval","pi_w_slope","pi_w_pval","beta","sigma","K","SLM_pval"])

host = sys.argv[1]
samples = config.Poyet_samples[host]
tp_matched = pd.read_pickle(f"{config.metadata_directory}Poyet_collection_dates.pkl")

rel_ab = pd.read_csv(f"{species_dir}/relative_abundance.txt.bz2",sep="\t",index_col=0)
species_list = os.listdir(f"{config.analysis_directory}/pi/Poyet")
print(species_list)

for species in species_list:  
   
    snps_dir = f"{data_dir}snps/{species}"
    out_path = f"{config.analysis_directory}pi/Poyet/{species}"

    pi = pd.read_csv(f"{out_path}/{species}_pi.txt",index_col=0)
    fst = pd.read_csv(f"{out_path}/{species}_fst.txt",index_col=0)
 
    sx = [s for s in samples if s in pi.index]
    samples_host = tp_matched[sx].sort_values().index

    tp = tp_matched[samples_host].sort_values()
    tp = pd.to_datetime(tp).diff().dt.days.cumsum()
       
    tp_matched = pd.read_pickle(f"{config.metadata_directory}Poyet_collection_dates.pkl")
    samples_host = config.Poyet_samples[host]
    samples_all = list(pd.read_csv(f"{snps_dir}/snps_depth.txt.bz2",sep="\t",index_col=0, nrows=0))
    samples_host = [sample for sample in samples_host if sample in samples_all]
    samples_host = [sample for sample in samples_host if sample in pi.index]
    
    if "SRR9224093" in samples_host:
        samples_host.remove("SRR9224093")   
    
    tp_ordered = tp_matched.loc[samples_host].sort_values().index
    
            ## remove mislabelled sample in ae
    if "SRR9224093" in tp_ordered:
        tp_ordered.remove("SRR9224093")   
    
    dates_host = np.array(list(dates[host][samples_host]))

    if len(samples_host) > 8:
        
        tp.iloc[0] = 0
        
        tp_ordered = [elem for elem in tp_ordered if elem in pi.index]

        L = list(rel_ab[tp_ordered].loc[species])          
            
        y_pi_b = list(pi.loc[tp_ordered,tp_ordered].loc[tp_ordered[0]])
        
        X = dates_host
        
        slopes_b = []
        for i in range(10000):
            pi_p = np.random.permutation(y_pi_b[1:])
            slopes_b.append(np.polyfit(X[1:], pi_p, 1)[0])
        
        true_slope_b = np.polyfit(X[1:], y_pi_b[1:], 1)[0]
        
        y_pi_w = list(np.diag(pi.loc[tp_ordered,tp_ordered]))
        slopes_w = []
        for i in range(10000):
            pi_p = np.random.permutation(y_pi_w)
            slopes_w.append(np.polyfit(X, pi_p, 1)[0])
        
        true_slope_w = np.polyfit(X, y_pi_w, 1)[0]
        
        species_df = pd.DataFrame(dates_host,index=samples_host,columns=["Date_Diffs"]).astype(int)
        
        SLM_results = species_SLM_test.run_SLM(species, host)
        
        out_row_dic = SLM_results
        
        out_row_dic["pi_b_slope"] = true_slope_b   
        
        out_row_dic["pi_b_pval"] = 1*(true_slope_b < slopes_b).sum()/len(slopes_b)
        
        out_row_dic["pi_w_slope"] = true_slope_w 
        
        out_row_dic["pi_w_pval"] = 1*(true_slope_w < slopes_w).sum()/len(slopes_w)
        
        out_df.loc[species] = out_row_dic
        
        sys.stderr.write(f"{species} complete \n \n \n")
        
        print(out_df)
        
out_df.to_csv(f"{config.analysis_directory}SLM_pi_fst_tables/SLM_pi_fst_tables_{host}.csv")