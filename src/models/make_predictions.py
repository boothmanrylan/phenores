import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import LabelBinarizer
from models import NeuralNetworkPipeline, SVMPipeline

def main():
    with open(snakemake.input[0], 'rb') as f:
        data = pickle.load(f)
        n_features = data.shape[1]
    with open(snakemake.input[1], 'rb') as f:
        target = pickle.load(f)
        n_classes = np.unique(target).shape[0]

    model = snakemake.wildcards.model

    if model == 'NN':
        encoder = LabelBinarizer()
        encoder.fit(target)
        target = encoder.transform(target)
        classifier = NeuralNetworkPipeline(n_classes, n_features)
    else:
        classifier = SVMPipeline(n_classes, n_features)

    predictions = cross_val_predict(classifier, data, target)

    if model == 'NN':
        target = encoder.inverse_transform(target)
        predictions = encoder.inverse_transform(predictions)

    label = snakemake.wildcards.label
    cols = ['Drug','Genome','True Value','{} {} Prediction'.format(model,label)]
    output = pd.DataFrame(index=np.arange(target.shape[0]), columns=cols)

    drug = snakemake.wildcards.drug
    for index, value in enumerate(predictions):
        output.loc[index] = [drug, predictions[index], value, target[index]]

    with open(snakemake.output[0], 'wb') as f:
        pickle.dump(output, f)

if __name__ == "__main__":
    main()
