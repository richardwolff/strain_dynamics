import pandas as pd
import numpy as np
import figure_utils
import matplotlib.pyplot as plt
import config
import pickle
import sys
from matplotlib import rc
rc('text', usetex=True)
SMALL_SIZE=15
MEDIUM_SIZE=20
LARGE_SIZE=25
rc('legend', fontsize=MEDIUM_SIZE)
rc('axes', labelsize=LARGE_SIZE)    # fontsize of the x and y labels
rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
rc('ytick', labelsize=MEDIUM_SIZE)

plt.style.use("ggplot")

analysis_dir = config.analysis_directory

good_species = {}

good_species["am"] = ["Bacteroides_fragilis_54507","Bacteroides_ovatus_58035","Bacteroides_vulgatus_57955",
               "Bacteroides_xylanisolvens_57185","Barnesiella_intestinihominis_62208","Eubacterium_eligens_61678",
               "Eubacterium_rectale_56927"]

good_species["an"] = ["Bacteroides_intestinalis_61596","Bacteroides_ovatus_58035","Bacteroides_uniformis_57318",
                      "Bacteroides_vulgatus_57955","Eubacterium_rectale_56927","Ruminococcus_bicirculans_59300",
                      "Sutterella_wadsworthensis_56828"]

good_species["ao"] = ["Bacteroides_thetaiotaomicron_56941","Bacteroides_xylanisolvens_57185","Bifidobacterium_adolescentis_56815",
                      "Eubacterium_eligens_61678","Eubacterium_rectale_56927","Faecalibacterium_prausnitzii_61481",
                      "Faecalibacterium_prausnitzii_62201"]

df = pd.read_csv("/u/home/r/rwolff/dbd/dbd_poyet/Poyet_polymorphism_rates_alpha_divs.csv",index_col=0)
df_gb = df.groupby(["species_name","subject_id"])


dates = pd.read_pickle("metadata/Poyet_collection_dates.pkl")
dates = pd.DataFrame(dates)
dates["Collection_Date"] = pd.to_datetime(dates.Collection_Date)

for host in ["am","an","ao"]:
    for species in good_species[host]:        
        try:
            
            pi_df = pd.DataFrame(df_gb.get_group((species,host))["polymorphism_rate"])
            pi_df["Collection_Date"] = dates["Collection_Date"]
            pi_df = pi_df.sort_values("Collection_Date")
            pi_df["Date_Diffs"] = pi_df["Collection_Date"].diff().dt.days
            pi_df["Date_Diffs"] = pi_df["Date_Diffs"].replace(0.0,.5)
            pi_df["Date_Diffs"][0] = 0.0

            timepoints = list(pi_df["Date_Diffs"].cumsum())
            pi = list(pi_df["polymorphism_rate"])
            
            species_name = figure_utils.get_pretty_species_name(species)
            
            fig,ax = plt.subplots(figsize=(12,8))
            ax.plot(timepoints,pi,'o');
            ax.semilogy()
            ax.set_ylim([1e-6,1e-1])
            ax.set_ylabel(r"$\pi$",rotation=0,size=25)
            ax.set_xlabel("Timepoint",size=25)
            ax.set_title(f"{species_name}, {host}")
            fig.savefig(f"{analysis_dir}temporal_pi/{host}/{species}.png")
            
            
        except:
            pass
    sys.stderr.write(f"{host} processed \n")