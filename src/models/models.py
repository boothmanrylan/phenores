import numpy as np
import pandas as pd
import pickle
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import RepeatedKFold, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelBinarizer, MinMaxScaler
from sklearn.externals import joblib
from sklearn.svm import SVC
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Conv1D, Flatten, Dense


class AddDimension(BaseEstimator, TransformerMixin):
    """
    Pipeline Class that adds a dimension to the X data.
    """
    def fit(self, x, y=None):
        return self

    def transform(self, x):
        x_prime = x.reshape(x.shape[0], x.shape[1], 1)
        return x_prime


class SelectionWrapper(BaseEstimator, TransformerMixin):
    """
    Pipeline Class that wraps SelectKBest to allow for target labels to be
    preprocessed/onehotencoded when using a neural net.
    """
    def __init__(self, k, classes, encode):
        """
        Args:
            k (int):        How many features to leave after feature selection.
            classes (int):  How many classes are in the data.
            encode (bool):  If True, apply inverse_transform to target labels
                            before feature selection.
        """
        self.encode = encode
        self.encoder = LabelBinarizer()
        self.classes = classes
        self.fit_to = np.arange(self.classes).reshape(self.classes, 1)
        self.encoder.fit(self.fit_to)
        self.k = k

    def fit(self, x, y=None):
        self.selector = SelectKBest(f_classif, k=self.k)

        if self.encode:
            labels = self.encoder.inverse_transform(y)
        else:
            labels = y

        self.selector.fit(x, labels)
        return self

    def transform(self, x):
        x_prime = self.selector.transform(x)
        return x_prime


def create_nn(classes=None, features=None):
    model = Sequential()
    model.add(Conv1D(filters=10,
                     kernel_size=3,
                     activation='relu',
                     input_shape=(features, 1)))
    model.add(Flatten())
    model.add(Dense(classes, activation='softmax'))
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def NeuralNetworkPipeline(n_classes, n_features):
    steps = [('selection', SelectionWrapper(n_features, n_classes, True)),
             ('scaling', MinMaxScaler(feature_range=(-1,1))),
             ('change_dims', AddDimension()),
             ('model', KerasClassifier(create_nn, epochs=50,
                                       batch_size=10, verbose=0,
                                       classes=n_classes,
                                       features=n_features))]
    classifier = Pipeline(steps)
    return classifier


def SVMPipeline(n_classes, n_features):
    steps = [('selection', SelectionWrapper(n_features, n_classes, False)),
             ('scaling', MinMaxScaler(feature_range=(-1,1))),
             ('model', SVC(kernel='linear'))]
    classifier = Pipeline(steps)
    return classifier
