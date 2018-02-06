import pickle
from sklearn.svm import SVC
from sklearn.externals import joblib

with open(snakemake.input[0], 'rb') as f:
    x_train = pickle.load(f)
with open(snakemake.input[1], 'rb') as f:
    y_train = pickle.load(f)

model = SVC(kernel='linear')

model.fit(x_train, y_train)

joblib.dump(model, snakemake.output[0])

