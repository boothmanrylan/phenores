import numpy as np
import pickle
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import MinMaxScaler, LabelBinarizer
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score
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


def create_nn(classes=None):
    model = Sequential()
    model.add(Conv1D(filters=10,
                     kernel_size=3,
                     activation='relu',
                     input_shape=(snakemake.config["select_k_features"], 1)))
    model.add(Flatten())
    model.add(Dense(classes, activation='softmax'))
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def main():
    with open(snakemake.input[0], 'rb') as f:
        data = pickle.load(f)
    with open(snakemake.input[1], 'rb') as f:
        targets = pickle.load(f)

    classes = np.unique(targets).shape[0]

    if snakemake.wildcards.MLtype == 'NN':
        encode = True
        encoder = LabelBinarizer()
        encoder.fit(targets)
        targets = encoder.transform(targets)
    else:
        encode = False

    k=snakemake.config["select_k_features"]

    steps = [('selection', SelectionWrapper(k, classes, encode)),
             ('scaling', MinMaxScaler(feature_range=(-1, 1)))]

    if snakemake.wildcards.MLtype == 'NN':
        steps.append(('change_dims', AddDimension()))
        steps.append(('model', KerasClassifier(create_nn, epochs=50,
                                               batch_size=10, verbose=0,
                                               classes=classes)))
    else:
        steps.append(('model', SVC(kernel='linear')))

    classifier = Pipeline(steps)

    predicted_vals = cross_val_predict(classifier, data, targets,
                                       cv=snakemake.config["cross_validations"])

    if snakemake.wildcards.MLtype == 'NN':
        targets = encoder.inverse_transform(targets)

    accuracy = accuracy_score(targets, predicted_vals)

    with open(snakemake.output[0], 'w') as f:
        f.write('{0}\n'.format(accuracy))


if __name__ == "__main__":
    main()
