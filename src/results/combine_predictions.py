import pickle
import pandas as pd
from functools import reduce

def main():
    frames = []
    for pkl in snakemake.input:
        with open(pkl, 'rb') as f:
            frames.append(pickle.load(f))

    cols=['Drug', 'Genome', 'True Value']
    output = reduce(lambda l, r: pd.merge(l, r, on=cols, how='outer'), frames)

    with open(snakemake.output[0], 'wb') as f:
        pickle.dump(output, f)

if __name__ == '__main__':
    main()
