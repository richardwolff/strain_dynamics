import sys
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import config
import seaborn as sns
import SLM_utils as slm
import plot_cluster_utils as pcu
plt.style.use("bmh")
import figure_utils
from scipy import stats 
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib import rc
rc('text', usetex=True)
SMALL_SIZE=15
MEDIUM_SIZE=20
rc('legend', fontsize=MEDIUM_SIZE)
rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
rc('xtick', labelsize=SMALL_SIZE)    # fonts

host = sys.argv[1]

for species in config.good_species[host]:
    output_directory = "/u/scratch/r/rwolff/strainfinder_input/%s" % host
    filename_prefix = "%s/%s" % (output_directory, species)
    snp_locations = pickle.load(open(filename_prefix+".strainfinder.locations.p",'rb'))
    snp_alignment = pd.read_pickle(filename_prefix+".strainfinder.p")
    snp_samples = pickle.load(open(filename_prefix+".strainfinder.samples.p",'rb'))
    snp_samples = [elem.decode("utf-8") for elem in snp_samples]


    dates = pd.read_pickle("metadata/Poyet_collection_dates.pkl")
    dates = pd.DataFrame(dates)
#dates = dates.loc[snp_samples]

    dates["Collection_Date"] = pd.to_datetime(dates.Collection_Date)
    
    outp=pd.read_pickle(f"~/diversity_ecology/analysis/clusters/{host}/{species}_strain_frequencies.pkl")

    cluster_As = snp_alignment[:,:,0].T
    cluster_Ds = (snp_alignment.sum(axis=2)).T

    good_inds = cluster_Ds.mean(axis=0) > 1
    good_inds = np.argwhere(good_inds == True).flatten()

    snp_map = pd.read_pickle(f"~/diversity_ecology/analysis/clusters/{host}/{species}_snp_map.pkl")
    freqs = pcu.get_clusters_snv_trajectories(snp_map)

    strain_df = pd.DataFrame(columns=outp.keys())
    for K in outp.keys():
        strain_freq_est = outp[K]["centroid"][0]
        strain_df[K] = strain_freq_est

    strain_df.index = snp_samples

    strain_df["Collection_Date"] = dates["Collection_Date"]
    strain_df["Collection_Date"] = dates["Collection_Date"]
    strain_df = strain_df.sort_values("Collection_Date")
    strain_df["Date_Diffs"] = strain_df["Collection_Date"].diff().dt.days
    strain_df["Date_Diffs"] = strain_df["Date_Diffs"].replace(0.0,.5)
    strain_df["Date_Diffs"][0] = 0.0
    strain_df["Order"] = range(strain_df.shape[0])
    sample_order = np.array(list(strain_df.loc[snp_samples]["Order"]))
    idx = np.empty_like(sample_order)
    idx[sample_order] = np.arange(len(sample_order))

    strain_freqs = pd.DataFrame(index=strain_df.index,columns=outp.keys())

    strain_freqs[list(outp.keys())] = strain_df[outp.keys()]
    out_strain_num = max(list(outp.keys())) + 1
    strain_freqs[out_strain_num] = 1 - strain_freqs.T.sum()
    times = np.array(list(strain_df["Date_Diffs"].cumsum()))

    rel_ab = pd.read_csv("/u/scratch/r/rwolff/merged_MIDAS_output/%s/species/relative_abundance.txt" % host,sep="\t",index_col=0)
    spec_rel_ab = rel_ab.loc[species]
    spec_rel_ab = spec_rel_ab.loc[strain_df.index]
    strain_total_freqs = (strain_freqs.T*spec_rel_ab).T

    species_name = figure_utils.get_pretty_species_name(species)

    fig,axs = plt.subplots(len(strain_total_freqs.mean()),figsize=(16,8))
    
    fig.suptitle(species_name,size=25)
    
    axs = axs.ravel()
    s = .17 

    i = 0
    for strain in strain_total_freqs.columns:
        K = (strain_total_freqs.mean()[strain]/(1-(s/2)))

        N = int(max(times))
        nse = stats.norm.rvs(size=N)

        x0 = strain_total_freqs.iloc[0][strain]
        x_list = []
        for j in range(1000):
            nse = stats.norm.rvs(size=N)
            x = slm.simulate_SLM(x0,19,s,K,N,nse)
            x_list.append(x)
    
        for k in range(1000):
    
            axs[i].plot(x_list[k],alpha=.1,color="red",zorder=1)
        
        axs[i].plot(times,strain_total_freqs[strain],color="k",linewidth=1)
        axs[i].scatter(times,strain_total_freqs[strain],s=50,color="k",edgecolor="k",zorder=2)

        axs[i].set_ylim([.1*min(strain_total_freqs[strain]),1]);

        axs[i].set_yscale("log")
        i+=1    
    
    fig.savefig(f"{config.analysis_directory}SLM_sim_figures/{host}/{species}.png")
                
    sys.stderr.write(f"{species_name} finished \n")