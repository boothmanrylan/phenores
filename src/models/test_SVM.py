import pickle
from datetime import datetime
from sklearn.externals import joblib

model = joblib.load(snakemake.input[0])

with open(snakemake.input[1], 'rb') as f:
    x_test = pickle.load(f)
with open(snakemake.input[2], 'rb') as f:
    y_test = pickle.load(f)

score = model.score(x_test, y_test)

with open(snakemake.output[0], 'a') as f:
    f.write("Time: {0!s}\n".format(datetime.now()))
    f.write("Accuracy: {0}\n".format(score))

