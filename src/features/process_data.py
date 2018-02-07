import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.feature_selection import f_classif, f_regression
from kmerprediction.feature_scaling import scale_to_range
from kmerprediction.feature_selection import select_k_best
import pickle

def convert_to_numbers(labels): # TODO: Get rid of this bin/regular aren't for regression and clean doesn't need it
    new_labels = np.zeros_like(labels, dtype='float64')
    for index, value in enumerate(labels):
        elem = str(value)
        if '=' in elem:
            new_labels[index] = float(elem[2:])
        elif '<' in elem:
            value = float(elem[1:])
            new_labels[index] = (0.99*value)
        elif '>' in elem:
            value = float(elem[1:])
            new_labels[index] = (1.01*value)
        elif elem == 'invalid': # TODO: This is not great
            new_labels[index] = -1
        else:
            new_labels[index] = float(elem)
    for index, value in enumerate(new_labels):
        if np.isnan(value): # TODO Also not great
            new_labels[index] = -1

    return new_labels

with open(snakemake.input[0], 'rb') as f:
    x_train = pickle.load(f)
with open(snakemake.input[1], 'rb') as f:
    y_train = pickle.load(f)
with open(snakemake.input[2], 'rb') as f:
    x_test = pickle.load(f)
with open(snakemake.input[3], 'rb') as f:
    y_test = pickle.load(f)

MLtype = snakemake.wildcards.MLtype # SVM or neural net
MLmethod = snakemake.wildcards.MLmethod # recursion or classification

if MLmethod == 'R': # Enforce numerical labels for recursion
    y_train = convert_to_numbers(y_train)
    y_test = convert_to_numbers(y_test)
    score_func = f_regression
else:
    if y_train.dtype == 'float64':
        y_train = y_train.astype('int')
    if y_test.dtype == 'float64':
        y_test = y_test.astype('int')
    score_func = f_classif

data = [x_train, y_train, x_test, y_test]
data, _ = select_k_best(data, None, score_func=score_func,  k=270)
data = scale_to_range(data)

x_train = data[0]
y_train = data[1]
x_test = data[2]
y_test = data[3]

if MLtype == 'NN': # Increase dimensionality of data for neural nets
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
    if MLmethod == 'C': # One-hot encode labels for classification
        all_labels = np.concatenate((y_train, y_test), axis=0)
        all_labels = all_labels.astype('str')
        encoder = LabelBinarizer()
        encoder.fit(all_labels)
        y_train = encoder.transform(y_train)
        y_test = encoder.transform(y_test)

with open(snakemake.output[0], 'wb') as f:
    pickle.dump(x_train, f)
with open(snakemake.output[1], 'wb') as f:
    pickle.dump(y_train, f)
with open(snakemake.output[2], 'wb') as f:
    pickle.dump(x_test, f)
with open(snakemake.output[3], 'wb') as f:
    pickle.dump(y_test, f)
