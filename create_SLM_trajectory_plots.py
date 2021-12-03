import sys
import config
import plot_SLM_trajectories_EM
import pandas as pd
import pickle

species_num = int(sys.argv[1]) - 1
#species = sys.argv[1]
host = sys.argv[2]

rel_ab = pd.read_csv(f"{config.data_directory}species/relative_abundance.txt.bz2",sep="\t",index_col=0)
psamps = config.Poyet_samples[host]
sorted_species = rel_ab[psamps].T.mean().sort_values(ascending=False)
sorted_species = list(sorted_species[sorted_species > 5*1e-3].index)
species = sorted_species[species_num]
#species = "Bacteroides_vulgatus_57955"
anal_dir = config.analysis_directory

out_dic = plot_SLM_trajectories_EM.run_SLM(species, host)
with open(f"{anal_dir}SLM_trajectory_figures/{host}/{species}/{species}_out_dic.pkl","wb") as f:
    pickle.dump(out_dic,f,protocol=2)
    
    