import numpy as np
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv1D
from keras.utils import to_categorical

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
model.add(Dense(y_train.shape[1], activation='softmax'))
model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=50, batch_size=10)

model.save(snakemake.output[0])
