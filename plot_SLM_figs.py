import sys
import pickle
import matplotlib.pyplot as plt
import numpy as np
import config
import seaborn as sns
import SLM_utils as slm
plt.style.use('ggplot')

#host = sys.argv[1]

good_species = {}

good_species["am"] = ["Bacteroides_fragilis_54507","Bacteroides_ovatus_58035","Bacteroides_vulgatus_57955",
               "Bacteroides_xylanisolvens_57185","Barnesiella_intestinihominis_62208","Eubacterium_eligens_61678",
               "Eubacterium_rectale_56927"]

good_species["an"] = ["Bacteroides_intestinalis_61596","Bacteroides_ovatus_58035","Bacteroides_uniformis_57318",
                      "Bacteroides_vulgatus_57955","Eubacterium_rectale_56927","Ruminococcus_bicirculans_59300",
                      "Sutterella_wadsworthensis_56828"]

#good_species["ae"] = 

good_species["ao"] = ["Bacteroides_thetaiotaomicron_56941","Bacteroides_xylanisolvens_57185","Bifidobacterium_adolescentis_56815",
                      "Eubacterium_eligens_61678","Eubacterium_rectale_56927","Faecalibacterium_prausnitzii_61481",
                      "Faecalibacterium_prausnitzii_62201"]

max_succ_times = 3

mean_abs = []
mean_diffs = []
mean_ratios = []

for host in good_species.keys():
    for species in good_species[host]:
        
        strain_total_freqs,strain_df = slm.calculate_strain_abundances(species, host)
        freq_diffs = slm.get_freq_diffs(strain_total_freqs, strain_df, max_succ_times)

        x,y = slm.calculate_mean_freq_diffs(freq_diffs, strain_total_freqs)
        mean_abs.extend(x)
        mean_diffs.extend(y)
        
        freq_ratios = slm.get_freq_ratios(strain_total_freqs, strain_df, max_succ_times)
        x,y = slm.calculate_mean_freq_ratios(freq_ratios, strain_total_freqs)
        
        mean_ratios.extend(y)
        
        sys.stderr.write(f"{species} finished \n")
    sys.stderr.write(f"{host} finished \n")
    
fig,axs = plt.subplots(2,figsize=(8,10),sharex=True)
axs[0].plot(mean_abs,mean_diffs,'.');
axs[0].set_ylabel(r"$\langle | x(t+\delta t) - x(t) | \rangle$",size=20)

#axs[0].set_xlabel(r"$\langle x(t) \rangle $",size=20)
axs[0].loglog()
axs[0].set_ylim([1e-4,1e-1])
#axs[0].set_xlim([1e-3,1e0])
#fig.savefig(f"{config.analysis_directory}/mean_diffs_fig.png")

#fig,ax = plt.subplots(figsize=(12,8))
axs[1].plot(mean_abs,mean_ratios,'.');
axs[1].set_ylabel(r"$\langle | \frac{x(t+\delta t)}{x(t)} | \rangle$",size=20)
axs[1].set_xlabel(r"$\langle x(t) \rangle $",size=20)
axs[1].loglog()
axs[1].set_ylim([1e-2,1e2])
axs[1].set_xlim([1e-3,1e0])
fig.savefig(f"{config.analysis_directory}/SLM_ratios_fig.png")
