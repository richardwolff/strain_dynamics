import sys
import os
import bz2
import parse_midas_data

#if len(sys.argv) > 1:
#    species_name=sys.argv[1]
#else:
#    species_name=parse_midas_data.debug_species_name

species_name = sys.argv[1]
#host = sys.argv[2]
sys.stderr.write("Calculating pvalues for %s...\n" % species_name)

output_filename = "%s/snps/%s/annotated_snps.txt.bz2" % (parse_midas_data.data_directory, species_name)

os.system('python %spipe_midas_data.py %s | %sannotate_pvalue --disabled | bzip2 -c > %s' % (parse_midas_data.scripts_directory, species_name, parse_midas_data.scripts_directory, output_filename) )  
 
