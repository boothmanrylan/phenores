import pickle
from datetime import datetime
import numpy as np
from sklearn.externals import joblib
from keras.utils import to_categorical
from keras.models import load_model
from kmerprediction.utils import same_shuffle

def test_NN(x_test, y_test, model_in):
    model = load_model(model_in)
    results = np.zeros(snakemake.config["validation_runs"], dtype='float64')
    for i in range(snakemake.config["validation_runs"]):
        loss, accuracy = model.evaluate(x_test, y_test)
        results[i] = accuracy
    return np.mean(results, axis=0)

def test_SVM(x_test, y_test, model_in):
    model = joblib.load(model_in)
    results = np.zeros(snakemake.config["validation_runs"], dtype='float64')
    for i in range(snakemake.config["validation_runs"]):
        score = model.score(x_test, y_test)
        results[i] = score
    return np.mean(results, axis=0)

if __name__ == "__main__":
    if snakemake.wildcards.MLtype == 'SVM':
        function = test_SVM
    else:
        function = test_NN

    results = np.zeros(snakemake.config["train_splits"] * snakemake.config["runs"],
                       dtype='float64')

    for ts in range(snakemake.config["train_splits"]):
        with open(snakemake.input[ts], 'rb') as f:
            test_data = pickle.load(f)
            x_test = test_data[0]
            y_test = test_data[1]
        for r in range(snakemake.config["runs"]):
            x_test, y_test = same_shuffle(x_test, y_test)
            x_test = np.asarray(x_test) # Necessarry because same shuffle returns lists
            y_test = np.asarray(y_test)
            index = snakemake.config["train_splits"] + (ts*snakemake.config["runs"]) + r
            model_in = snakemake.input[index]

            results[index - snakemake.config["train_splits"]] = function(x_test, y_test, model_in)

    with open(snakemake.output[0], 'a') as f:
        f.write("{0}\n".format(str(datetime.now())))
        f.write("Train Splits: {0}\n".format(snakemake.config["train_splits"]))
        f.write("Runs: {0}\n".format(snakemake.config["runs"]))
        f.write("Avg. Accuracy: {0}\n".format(np.mean(results, axis=0)))
