import os
from kmerprediction import kmer_counter
all_files = []
directory = snakemake.input[0]
files = [directory + x for x in os.listdir(directory)]
all_files.extend(files)
kmer_counter.count_kmers(snakemake.config["k"], snakemake.config["l"], files,
                         snakemake.output[0], True)

