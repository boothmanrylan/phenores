import os
from kmerprediction import kmer_counter

directory = snakemake.input[0]

files = [directory + x for x in os.listdir(directory)]

kmer_counter.count_kmers(snakemake.config["k"], snakemake.config["l"], files,
                         snakemake.output[0], True)

