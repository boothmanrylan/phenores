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


def create_pipeline(MLtype, num_features, num_classes):
    if MLtype == 'NN':
        encoder = LabelBinarizer()
        encoder.fit(target)
        target = encoder.transform(target)
        encode = True
    else:
        encode = False

    steps = [('selection', SelectionWrapper(num_features, num_classes, encode)),
             ('scaling', MinMaxScaler(feature_range=(-1, 1)))]

    if MLtype == 'NN':
        steps.append(('change_dims', AddDimension()))
        steps.append(('model', KerasClassifier(create_nn, epochs=50,
                                               batch_size=10, verbose=0,
                                               classes=classes, features=num_features)))
    else:
        steps.append(('model', SVC(kernel='linear')))

    classifier = Pipeline(steps)
    return classifier


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


def main():
    with open(snakemake.input[0], 'rb') as f:
        data = pickle.load(f)
    with open(snakemake.input[1], 'rb') as f:
        target = pickle.load(f)
        classes = np.unique(target).shape[0]

    model_type = snakemake.wildcards.MLtype
    drug = snakemake.wildcards.drug
    label_encoding = snakemake.wildcards.label
    output_file = snakemake.output[0]
    splits = snakemake.config['n_splits']
    repeats = snakemake.config['n_repeats']
    k = snakemake.config["select_k_features"]

    classifier = make_pipeline(model_type, k, classes)

    if 'results' in output_file:
        rkf = RepeatedKFold(n_splits=splits, n_repeats=repeats)

        scores = cross_val_score(classifier, data, target, cv=rkf)

        output = pd.DataFrame(columns=['Drug', 'Model Type', 'Label Encoding',
                                       'Mean Accuracy', 'Std. Deviation',
                                       '# of Runs'])

        output.loc[0] = [drug, model_type, label_encoding, np.mean(scores),
                         np.std(scores), scores.shape[0]]
    else:
        predictions = cross_val_predict(classifier, data, target, cv=10)

        if MLtype == 'NN':
            target = encoder.inverse_transform(target)
            predictions = encoder.inverse_transform(predictions)

        output = pd.DataFrame(index=np.arange(genomes.shape[0]),
                              columns=['Drug', 'Genome', 'True Value',
                                       '{} {} Prediction'.format(MLtype, label)])

        count = 0
        for index, value in enumerate(predictions):
            output.loc[count] = [drug, genomes[index], value, target[index]]
            count += 1

    with open(output_file, 'wb') as f:
        pickle.dump(output, f)

if __name__ == "__main__":
    main()
