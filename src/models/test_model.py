import pickle
from datetime import datetime
import numpy as np
from sklearn.externals import joblib
from keras.utils import to_categorical
from keras.models import load_model

def test_NN(x_test_file, y_test_file, model_in, results_file):
    with open(x_test_file, 'rb') as f:
        x_test = pickle.load(f)
    with open(y_test_file, 'rb') as f:
        y_test = pickle.load(f)

    model = load_model(model_in)

    loss, accuracy = model.evaluate(x_test, y_test)
    return accuracy

def test_SVM(x_test_file, y_test_file, model_in, results_file):
    with open(x_test_file, 'rb') as f:
        x_test = pickle.load(f)
    with open(y_test_file 'rb') as f:
        y_test = pickle.load(f)

    model = joblib.load(model_in)

    score = model.score(x_test, y_test)
    return score

if __name__ == "__main__":
    if snakemake.wildcards.MLtype == 'SVM':
        function = train_SVM
    else:
        function = train_NN

    offset = len(snakemake.config["train_splits"]) * 2

    results = np.zeros(k+K, dtype='float64')

    for k in snakemake.config["train_splits"]:
        i = k*2
        x_test_file = snakemake.input[i]
        y_test_file = snakemake.input[i+1]
        for K in snakemake.config["train_runs"]:
            index = offset + k + K
            model_in = snakemake.input[index]
            results[index-offset] = function(x_test, y_test, model_in)
    with open(snakemake.output[0], 'a') as f:
        f.write("{0}\n".format(str(datetime.now())))
        f.write("Train Splits: {0}\n".format(len(snakemake.config["train_splits"])))
        f.write("Runs: {0}\n".format(len(snakemake.config["train_runs"])))
        f.write("Accuracy: {0}\n".format(np.mean(results, axis=0)))
