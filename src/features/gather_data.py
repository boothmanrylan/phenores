import pickle
import unicodedata
import random
import os
import pandas as pd
import numpy as np
from kmerprediction.kmer_counter import get_counts, get_kmer_names
from sklearn.externals import joblib

if __name__ == "__main__":
    directory = snakemake.input[0]
    db = snakemake.input[1]
    metadata = snakemake.input[2]

    data_file = snakemake.output[0]
    target_file = snakemake.output[1]

    metadata = joblib.load(metadata)

    files = [directory + x for x in os.listdir(directory)]
    data = pd.DataFrame(get_counts(files, db), columns=get_kmer_names(db))

    labels = [x.replace(directory, '').replace('.fasta', '') for x in files]

    drug = snakemake.wildcards.drug
    mic_vals = metadata[drug]

    targets = np.zeros(len(labels), dtype='<U7')
    for index, value in enumerate(labels):
        label = mic_vals.loc[mic_vals.index == value].values
        targets[index] = label[0]

    with open(data_file, 'wb') as f:
        pickle.dump(data, f)
    with open(target_file, 'wb') as f:
        pickle.dump(targets, f)

