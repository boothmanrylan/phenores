import pandas as pd
import numpy as np
import pickle

def main():
    frames = []
    for pkl in snakemake.input:
        with open(pkl, 'rb') as f:
            frames.append(pickle.load(f))

    output = pd.concat(frames, ignore_index=True)

    with open(snakemake.output[0], 'wb') as f:
        pickle.dump(output, f)

if __name__ == "__main__":
    main()
