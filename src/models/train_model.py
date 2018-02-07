import pickle
import numpy as np
from sklearn.svm import SVR
from sklearn.svm import SVC
from sklearn.externals import joblib
from keras.wrappers.scikit_learn import KerasRegressor
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv1D
from keras.utils import to_categorical

def train_SVM_C(x_train_file, y_train_file, model_out):
    with open(x_train_file, 'rb') as f:
        x_train = pickle.load(f)
    with open(y_train_file, 'rb') as f:
        y_train = pickle.load(f)

    model = SVC(kernel='linear')

    model.fit(x_train, y_train)

    joblib.dump(model, model_out)

def train_SVM_R(x_train_file, y_train_file, model_out):
    with open(x_train_file, 'rb') as f:
        x_train = pickle.load(f)
    with open(y_train_file, 'rb') as f:
        y_train = pickle.load(f)

    model = SVR(kernel='linear')

    model.fit(x_train, y_train)

    joblib.dump(model, model_out)

def train_NN_C(x_train_file, y_train_file, model_out):
    with open(x_train_file, 'rb') as f:
        x_train = pickle.load(f)
    with open(y_train_file, 'rb') as f:
        y_train = pickle.load(f)

    model = Sequential()
    model.add(Conv1D(filters=10,
                     kernel_size=3,
                     activation='relu',
                     input_shape=x_train.shape[1:]))
    model.add(Flatten())
    model.add(Dense(y_train.shape[1], activation='softmax'))
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=50, batch_size=10)

    model.save(model_out)

def train_NN_R(x_train_file, y_train_file, model_out):
    with open(x_train_file, 'rb') as f:
        x_train = pickle.load(f)
    with open(y_train_file, 'rb') as f:
        y_train = pickle.load(f)

    model = Sequential()
    model.add(Conv1D(filters=10,
                     kernel_size=3,
                     activation='relu',
                     input_shape=x_train.shape[1:]))
    model.add(Flatten())
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(optimizer='adam',
                  loss='mean_squared_error',
                  metrics=['accuracy'])
    y_train = np.asarray(y_train)

    model.fit(x_train, y_train, epochs=50, batch_size=10)

    model.save(model_out)

if __name__ == "__main__":
    if snakemake.wildcards.MLtype == 'NN':
        if snakemake.wildcard.MLmethod == 'R':
            function = train_NN_R
        else:
            function = train_NN_C
    else:
        if snakemake.wildcards.MLmethod == 'R':
            function = train_SVM_R
        else:
            function = train_SVM_C

    for k in snakemake.config["train_splits"]:
        i = k*2
        x_train_file = snakemake.input[i]
        y_train_file = snakemake.input[i+1]
        for K in snakemake.config["train_runs"]:
            model_out = snakemake.output[k+K]
            function(x_train_file, y_train_file, model_out)
