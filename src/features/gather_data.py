import pickle
import unicodedata
import random
import os
import pandas as pd
import numpy as np
from kmerprediction.kmer_counter import get_counts
from sklearn.externals import joblib

directory = snakemake.input[0]
database = snakemake.input[1]
metadata = snakemake.input[2]

def gather_data(x_train_file, y_train_file, x_test_file, y_test_file):
    metadata = joblib.load(metadata)

    full_filepaths = [directory + x for x in os.listdir(directory)]
    random.shuffle(full_filepaths)

    cutoff = int(float(snakemake.config["train_size"]) * len(full_filepaths))
    train_files = full_filepaths[:cutoff]
    test_files = full_filepaths[cutoff:]

    x_train = get_counts(train_files, database)
    x_test = get_counts(test_files, database)

    train_files = [x.replace(directory, '').replace('.fasta', '') for x in train_files]
    test_files = [x.replace(directory, '').replace('.fasta', '') for x in test_files]

    drug = snakemake.wildcards.drug

    y_train = []
    for x in train_files:
        label = metadata[drug][x]
        y_train.append(label)
    y_train = np.asarray(y_train)

    y_test = []
    for x in test_files:
        label = metadata[drug][x]
        y_test.append(label)
    y_test = np.asarray(y_test)

    with open(x_train_file, 'wb') as f:
        pickle.dump(x_train, f)
    with open(y_train_file, 'wb') as f:
        pickle.dump(y_train, f)
    with open(x_test_file, 'wb') as f:
        pickle.dump(x_test, f)
    with open(y_test_file, 'wb') as f:
        pickle.dump(y_test, f)

if __name__ == "__main__":
    for k in snakemake.config["train_splits"]:
        i = k*4
        x_train_file = snakemake.output[i]
        y_train_file = snakemake.output[i+1]
        x_test_file = snakemake.output[i+2]
        y_test_file = snakmake.output[i+3]
        gather_data(x_train_file, y_train_file, x_test_file, y_test_file)
