###############################
#
# Rest of script begins here
#
################################
import matplotlib  
matplotlib.use('Agg') 
import pylab
import numpy
import sys
from math import log10
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl
from numpy.random import binomial, random_sample
import numpy as np
import bz2
import parse_midas_data
import parse_timecourse_data
import matplotlib
import matplotlib.pyplot as plt
import calculate_preexisting_snps
import cluster_utils
import os
import cPickle
import config
import random
from math import fabs
mpl.rcParams['font.size'] = 7
mpl.rcParams['lines.linewidth'] = 1
mpl.rcParams['legend.frameon']  = False
mpl.rcParams['legend.fontsize']  = 'small'

min_coverage = 1

species_name = sys.argv[1]
host = sys.argv[2]

output_directory = "/u/scratch/r/rwolff/strainfinder_input/Poyet/%s" % host
filename_prefix = "%s/%s" % (output_directory, species_name)

# Load location object
snp_locations = cPickle.load(open(filename_prefix+".strainfinder.locations.p",'rb'))
snp_alignment = cPickle.load(open(filename_prefix+".strainfinder.p",'rb'))

cluster_filename = filename_prefix+".clusters.p"

cluster_As = snp_alignment[:,:,0].T
cluster_Ds = (snp_alignment.sum(axis=2)).T
cluster_Fs = (cluster_As/cluster_Ds)

num_sites = numpy.array([sum(1*(numpy.logical_and(e > 0, e < 1)))/(1.0*cluster_Fs.shape[1]) for e in cluster_Fs])

## only keep snps appearing in > 10% of timepoints
good_sites = numpy.array([e[0] for e in numpy.argwhere(num_sites > .1)])
print(good_sites)

cluster_As = snp_alignment[:,good_sites,0].T
cluster_Ds = (snp_alignment[:,good_sites,:].sum(axis=2)).T

pylab.figure(figsize=(7,4))
fig = pylab.gcf()

outer_grid  = gridspec.GridSpec(3,2, height_ratios=[1,1,1], width_ratios=[2,1], hspace=0.15)

observed_axis = plt.Subplot(fig, outer_grid[0,0])
fig.add_subplot(observed_axis)
observed_axis.set_ylim([0,1])
observed_axis.set_xlim([-5,160])
observed_axis.set_ylabel('Allele\nfrequency')
observed_axis.spines['top'].set_visible(False)
observed_axis.spines['right'].set_visible(False)
observed_axis.get_xaxis().tick_bottom()
observed_axis.get_yaxis().tick_left()
for snp_idx in xrange(0,cluster_As.shape[0]):
    
    alts = cluster_As[snp_idx,:]
    depths = cluster_Ds[snp_idx,:]
    freqs = alts*1.0/(depths+(depths==0))
    
    good_idxs = (depths>min_coverage)
    
    ## rtw: hack
    ts = numpy.asarray(range(len(depths)))
    #observed_axis.plot(ts[good_idxs],freqs[good_idxs],'-',alpha=0.1,linewidth=0.25)

    
predicted_axis = plt.Subplot(fig, outer_grid[1,0])
fig.add_subplot(predicted_axis)
predicted_axis.set_ylim([0,1])
predicted_axis.set_xlim([-5,160])

predicted_axis.set_ylabel('Allele\nfrequency')
predicted_axis.spines['top'].set_visible(False)
predicted_axis.spines['right'].set_visible(False)
predicted_axis.get_xaxis().tick_bottom()
predicted_axis.get_yaxis().tick_left()

rest_axis = plt.Subplot(fig, outer_grid[2,0])
fig.add_subplot(rest_axis)
rest_axis.set_ylim([0,1])
rest_axis.set_xlim([-5,160])

rest_axis.set_ylabel('Allele\nfrequency')
rest_axis.spines['top'].set_visible(False)
rest_axis.spines['right'].set_visible(False)
rest_axis.get_xaxis().tick_bottom()
rest_axis.get_yaxis().tick_left()
rest_axis.set_xlabel('Time (days)')

distance_axis = plt.Subplot(fig, outer_grid[1,1])
fig.add_subplot(distance_axis)
sm = cmx.ScalarMappable(norm=colors.Normalize(vmin=0, vmax=0.4),cmap=cmx.get_cmap(name='Greys'))
sm.set_array([])
fig.colorbar(sm, ax=distance_axis)


def hex_to_rgb(value):
    #value = value.lstrip('#')
    #print(value)
    #lv = len(value)
    #return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))
    #return tuple(int(lv[i:i+2], 16) for i in (0, 2, 4))
    #return colors.to_hex(value)
    color = tuple([random.randint(0,256),random.randint(0,256),random.randint(0,256)])

    return color

def fraction_to_rgb(f,fmax=0.5):
	c = long((1-f/fmax)*255)
	return (c,c,c)
              
#cluster_map = cluster_utils.cluster_snps_by_distance(cluster_As, cluster_Ds,max_d=config.cluster_distance_threshold_barcodes)

#cluster_map = cluster_utils.cluster_snps_by_distance(cluster_As, cluster_Ds,max_d=config.cluster_distance_threshold_barcodes)
print(cluster_As/cluster_Ds)
cluster_map = cluster_utils.fast_cluster_snps_by_distance(cluster_As, cluster_Ds,max_d=config.cluster_distance_threshold_barcodes)

total_clustered_snvs = 0
total_fractional_clustered_snvs = 0
        
cluster_data = []

cluster_color_map = {}  
good_clusters = []         
cluster_ids = cluster_map.keys()
cluster_sizes = [len(cluster_map[cluster_id]['snps']) for cluster_id in cluster_ids]

sorted_cluster_sizes, sorted_cluster_ids = zip(*sorted(zip(cluster_sizes, cluster_ids), reverse=True))

output_strs = []
output_str = ", ".join(['Cluster', 'Contig', 'Location'])
output_strs.append(output_str)

cluster_freq_map = {}
cluster_color_map = {}

for cluster_id in sorted_cluster_ids:
    color = None
    avg_fs, total_Ds = cluster_map[cluster_id]['centroid']
            
    good_idxs = (total_Ds>0)
            
    if good_idxs.sum()<1.5:
        continue
    print(avg_fs[:10])
    
    ## hack RTW
    #ts = range(len(masked_avg_fs))
    masked_times = ts[good_idxs]
    masked_avg_fs = avg_fs[good_idxs]
            
    cluster_size = len(cluster_map[cluster_id]['snps'])
    fractional_size = cluster_size*1.0/len(cluster_As)     
    
    sys.stderr.write("Cluster %d (n=%d SNVs, %g)\n" % (cluster_id, cluster_size, fractional_size)) 
    
    #if cluster_size>=100:
    if fractional_size>0.1:
        keeper=True
    else:
        keeper=False
        for snp_idx, flip in sorted(cluster_map[cluster_id]['snps']):
            good_idxs = cluster_Ds[snp_idx,:]>0
            masked_times = ts[good_idxs]
            masked_As = cluster_As[snp_idx,good_idxs]
            masked_Ds = cluster_Ds[snp_idx,good_idxs]
            if flip:
                masked_As = masked_Ds-masked_As
            masked_freqs = masked_As*1.0/masked_Ds


    if not keeper: 
        continue
 
    line, = predicted_axis.plot([1],[-1],'.')
    color = pylab.getp(line,"color")
    print("%s is the color \n" % color)
    cluster_color_map[cluster_id] = color
    good_clusters.append(cluster_id)
    total_clustered_snvs += cluster_size
    total_fractional_clustered_snvs += fractional_size
                 
    cluster_freq_map[cluster_id] = (cluster_size,masked_times, masked_avg_fs)
    cluster_snp_locations = []  
    for snp_idx, flip in sorted(cluster_map[cluster_id]['snps']):
        good_idxs = cluster_Ds[snp_idx,:]>0
        masked_times = ts[:]
        masked_As = cluster_As[snp_idx,:]
        masked_Ds = cluster_Ds[snp_idx,:]
        
        contig,location,allele = snp_locations[snp_idx]        
                
        if flip:
            masked_As = masked_Ds-masked_As
            
            if allele=='A':
                allele='R'
            else:
                allele='A'
        
        masked_freqs = masked_As*1.0/masked_Ds
        
        cluster_data.append( ( contig, location, allele, cluster_id, masked_As, masked_Ds ) )
        cluster_snp_locations.append((contig, location))
    cluster_snp_locations.sort()
    for contig, location in cluster_snp_locations:
        output_str = ", ".join([str(color), str(contig), str(location)])
        output_strs.append(output_str)
        
sys.stderr.write("Total clustered SNVs: %d\n" % total_clustered_snvs)
sys.stderr.write("Total fraction clustered: %g\n" % total_fractional_clustered_snvs)     

filename = parse_midas_data.analysis_directory+('clusters/%s/%s_cluster_figure_clusters.txt' % (host,species_name))

file = open(filename,"w")
for output_str in output_strs:
    file.write(output_str)
    file.write("\n")
file.close()

snp_cluster_map = {}
for contig, location, allele, cluster_id, As, Ds in cluster_data:
    snp_cluster_map[(contig,location)] = (allele,cluster_id, As, Ds)
strain_freqs = parse_midas_data.analysis_directory+('clusters/%s/%s_strain_frequencies.pkl' % (host,species_name))

import pickle
with open(strain_freqs,"wb") as f:
    
    pickle.dump({good_cluster:cluster_map[good_cluster] for good_cluster in good_clusters},f)
snp_map = parse_midas_data.analysis_directory+('clusters/%s/%s_snp_map.pkl' % (host,species_name))

with open(snp_map,"wb") as f:
    
    pickle.dump(snp_cluster_map,f)
    
sys.stderr.write("Done!\n")

