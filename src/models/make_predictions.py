import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import MinMaxScaler, LabelBinarizer
from sklearn.externals import joblib

def main():
    with open(snakemake.input[0], 'rb') as f:
        data = pickle.load(f)
    with open(snakemake.input[1], 'rb') as f:
        target = pickle.load(f)

    MLtype = snakemake.wildcards.MLtype
    label = snakemake.wildcards.label
    drug = snakemake.wildcards.drug

    if MLtype == 'NN':
        encoder = LabelBinarizer()
        encoder.fit(targets)
        target = encoder.transform(target)

    classifier = joblib.load(snakemake.input[2])

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

    with open(snakemake.output[0], 'wb') as f:
        pickle.dump(output, f)

if __name__ == "__main__":
    main()
