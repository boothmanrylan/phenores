import pickle
import unicodedata
import random
import os
import pandas as pd
import numpy as np
from kmerprediction.kmer_counter import get_counts
from sklearn.externals import joblib

if __name__ == "__main__":
    directory = snakemake.input[0]
    database = snakemake.input[1]
    metadata = snakemake.input[2]

    data_file = snakemake.output[0]
    target_file = snakemake.output[1]

    metadata = joblib.load(metadata)

    files = [directory + x for x in os.listdir(directory)]
    data = get_counts(files, database)

    labels = [x.replace(directory, '').replace('.fasta', '') for x in files]

    drug = snakemake.wildcards.drug

    target = np.zeros(len(labels), dtype='str')
    for index, value in enumerate(labels):
        label = metadata[drug][value]
        target[index] = label

    with open(data_file, 'wb') as f:
        pickle.dump(data, f)
    with open(target_file, 'wb') as f:
        pickle.dump(target, f)

