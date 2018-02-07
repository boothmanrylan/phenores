import pickle
import numpy as np
from keras.wrappers.scikit_learn import KerasRegressor
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv1D
from sklearn.externals import joblib

with open(snakemake.input[0], 'rb') as f:
    x_train = pickle.load(f)
with open(snakemake.input[1], 'rb') as f:
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

model.save(snakemake.output[0])
