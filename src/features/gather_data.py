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

with open(snakemake.output[0], 'wb') as f:
    pickle.dump(x_train, f)
with open(snakemake.output[1], 'wb') as f:
    pickle.dump(y_train, f)
with open(snakemake.output[2], 'wb') as f:
    pickle.dump(x_test, f)
with open(snakemake.output[3], 'wb') as f:
    pickle.dump(y_test, f)
