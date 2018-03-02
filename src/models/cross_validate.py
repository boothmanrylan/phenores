import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedKFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler, LabelBinarizer
from sklearn.externals import joblib
from build_model import SelectionWrapper, AddDimension, create_nn

def main():
    with open(snakemake.input[0], 'rb') as f:
        data = pickle.load(f)
    with open(snakemake.input[1], 'rb') as f:
        target = pickle.load(f)

    model_type = snakemake.wildcards.MLtype
    drug = snakemake.wildcards.drug
    label_encoding = snakemake.wildcards.label

    if model_type == 'NN':
        encoder = LabelBinarizer()
        encoder.fit(target)
        target = encoder.transform(target)

    classifier = joblib.load(snakemake.input[2])

    splits = snakemake.config['n_splits']
    repeats = snakemake.config['n_repeats']
    rkf = RepeatedKFold(n_splits=splits, n_repeats=repeats)

    scores = cross_val_score(classifier, data, target, cv=rkf)

    output = pd.DataFrame(columns=['Drug', 'Model Type', 'Label Encoding',
                                   'Mean Accuracy', 'Std. Deviation',
                                   '# of Runs'])

    output.loc[0] = [drug, model_type, label_encoding, np.mean(scores),
                     np.std(scores), scores.shape[0]]

    with open(snakemake.output[0], 'wb') as f:
        pickle.dump(output, f)

if __name__ == "__main__":
    main()
