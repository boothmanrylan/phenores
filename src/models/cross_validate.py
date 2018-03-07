import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from models import NeuralNetworkPipeline, SVMPipeline

def main():
    with open(snakemake.input[0], 'rb') as f:
        data = pickle.load(f)
        n_features = data.shape[1]
    with open(snakemake.input[1], 'rb') as f:
        target = pickle.load(f)
        n_classes = np.unique(target).shape[0]

    model = snakemake.wildcards.model
    drug = snakemake.wildcards.drug
    label = snakemake.wildcards.label

    kf = KFold(n_splits=snakemake.config['n_splits'], shuffle=True)

    if model == 'NN':
        encoder = LabelBinarizer()
        encoder.fit(target)
        target = encoder.transform(target)
        classifier = NeuralNetworkPipeline(n_classes, n_features)
        scores = cross_val_score(classifier, data, target, cv=kf)
    else:
        scores, coefs = [], []
        for train, test in kf.split(data, target):
            classifier = SVMPipeline(n_classes, n_features)
            classifier.fit(data.as_matrix()[train], target[train])
            scores.append(classifier.score(data.as_matrix[test], target[test]))
            coef = np.absolute(clf.coef_)
            if coef.ndim > 1:
                coef = coef.sum(axis=1)
            coefs.append(coef)
        feature_coefs = pd.DataFrame(coefs, columns=data.columns)
        with open(snakemake.output[1], 'wb') as f:
            pickle.dump(feature_coefs, f)

    cols = ['Drug', 'Model Type', 'Label Encoding', 'Accuracy']
    index = np.arange(snakemake.config['n_splits'])
    output = pd.DataFrame(columns=cols, index=index)

    for index, value in enumerate(scores):
        output.loc[index] = [drug, model, label, value]

    with open(snakemake.output[0], 'wb') as f:
        pickle.dump(output, f)

if __name__ == "__main__":
    main()
