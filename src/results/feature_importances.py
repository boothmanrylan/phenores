import numpy as np
import pandas as pd
import pickle

def aggregation(input_series):
    output = sum([1/(2**x) for x in input_series.values])
    return output

def main():
    with open(snakemake.input[0], 'rb') as f:
        coefs = pickle.load(f)
    with open(snakemake.input[1], 'rb') as f:
        distribution = pickle.load(f)

    ranks = coefs.apply(np.argsort, axis=1)
    scores = ranks.apply(aggregation, axis=0)
    scores = scores.apply(lambda x: x/coefs.shape[0])

    for f in scores.index:
        distribution.loc[distribution['K-mer'] == f, 'Feature Importances'] = scores.loc[f]

    feature_importances = distribution.dropna(axis=0, how='any')

    with open(snakemake.output[0], 'wb') as f:
        pickle.dump(feature_importances, f)

if __name__ == "__main__":
    main()
