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


def create_pipeline(MLtype, num_features, num_classes):
    if MLtype == 'NN':
        encode = True
    else:
        encode = False

    steps = [('selection', SelectionWrapper(num_features, num_classes, encode)),
             ('scaling', MinMaxScaler(feature_range=(-1, 1)))]

    if MLtype == 'NN':
        steps.append(('change_dims', AddDimension()))
        steps.append(('model', KerasClassifier(create_nn, epochs=50,
                                               batch_size=10, verbose=0,
                                               classes=num_classes,
                                               features=num_features)))
    else:
        steps.append(('model', SVC(kernel='linear')))

    classifier = Pipeline(steps)
    return classifier


def cross_validate(MLtype, classifier, data, target):
    kf = KFold(n_splits=snakemake.config['n_splits'], shuffle=True)

    cols = ['Drug', 'Model Type', 'Label Encoding', 'Accuracy']
    output = pd.DataFrame(columns=cols, index=np.arange(n_splits))

    # If the model is an SVM output feature coeffcients and accuracies
    # If the model is an NN only output accuracies
    if MLtype == 'SVM':
        scores, coefs = [], []
        for train, test in rkf.split(data, target):
            clf = classifier
            clf.fit(data.as_matrix()[train], target[train])
            scores.append(clf.score(data.as_matrix()[test], target[test]))
            coef = clf.coef_

            if coef.ndim > 1:
                coef = coef.sum(axis=1)

            coefs.append(np.absolute(coef))
        feature_coefs = pd.DataFrame(coefs, columns=data.columns)
    else:
        scores = cross_val_score(classifier, data.as_matrix(), target, cv=kf)
        feature_coefs = None

    drug = snakemake.wildcards.drug
    label = snakemake.wildcard.label
    for index, value in enumerate(scores):
        output.loc[index] = [drug, model_type, label, value]

    return output, feature_coefs


def make_predictions(MLtype, classifier, data, target)
    predictions = cross_val_predict(classifier, data, target, cv=3)

    if MLtype == 'NN':
        encoder = LabelBinarizer()
        encoder.fit(target)
        target = encoder.transform(target)
        target = encoder.inverse_transform(target)
        predictions = encoder.inverse_transform(predictions)

    cols = ['Drug','Genome','True value','{} {} Prediction'.format(MLtype,label)]
    output = pd.DataFrame(columns=cols, index=np.arange(genomes.shape[0]))

    drug = snakemake.wildcards.drug
    for index, value in enumerate(predictions):
        output.loc[index] = [drug, genomes[index], value, target[index]]

    return output


def main():
    with open(snakemake.input[0], 'rb') as f:
        data = pickle.load(f)
    with open(snakemake.input[1], 'rb') as f:
        target = pickle.load(f)
        classes = np.unique(target).shape[0]

    mtype = snakemake.wildcards.MLtype
    k = snakemake.config["select_k_features"]
    clf = create_pipeline(model_type, k, classes)

    if snakemake.wildcards.run_type == 'results':
        output, feature_coefs = cross_validate(mtype, clf, data, target)
    else:
        output = make_predictions(mtype, clf, data, target)
        feature_coefs = None

    with open(snakemake.output[0], 'wb') as f:
        pickle.dump(output, f)
    with open(snakemake.output[1], 'wb') as f:
        pickle.dump(feature_coefs, f)

if __name__ == "__main__":
    main()
