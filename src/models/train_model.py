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
from kmerprediction.utils import same_shuffle

def train_SVM_C(x_train, y_train, model_out):
    model = SVC(kernel='linear')

    model.fit(x_train, y_train)
    joblib.dump(model, model_out)

def train_NN_C(x_train, y_train, model_out):
    model = Sequential()
    model.add(Conv1D(filters=10,
                     kernel_size=3,
                     activation='relu',
                     input_shape=x_train.shape[1:]))
    model.add(Flatten())
    model.add(Dense(y_train.shape[1], activation='softmax'))
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=50, batch_size=10, verbose=0)
    model.save(model_out)

if __name__ == "__main__":
    if snakemake.wildcards.MLtype == 'NN':
        function = train_NN_C
    else:
        function = train_SVM_C
    print(snakemake.output)
    for ts in range(snakemake.config["train_splits"]):
        with open(snakemake.input[ts], 'rb') as f:
           train_data = pickle.load(f)
           x_train = train_data[0]
           y_train = train_data[1]
        for r in range(snakemake.config["runs"]):
            x_train, y_train = same_shuffle(x_train, y_train)
            x_train = np.asarray(x_train) # Necessary because same_shuffle returns lists
            y_train = np.asarray(y_train)
            index = (ts*snakemake.config["runs"]) + r
            print(index)
            model_out = snakemake.output[index]
            function(x_train, y_train, model_out)
