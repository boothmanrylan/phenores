import matplotlib

matplotlib.use('agg')

import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import numpy as np
import pandas as pd

def main():
    with open(snakemake.input[0], 'rb') as f:
        data = pickle.load(f)

    sns.set(font_scale=1.5)

    g = sns.factorplot(x='Model Type', y='Accuracy', hue='Label Encoding',
                       col='Drug', kind='box', data=data, legend=False,
                       size=6, fliersize=10, linewidth=2, col_wrap=4,
                       medianprops={'color': 'black', 'linewidth': 3,
                                    'solid_capstyle': 'butt'})

    g.add_legend(title='Label Encoding')
    plt.subplots_adjust(top=0.9)
    g.fig.suptitle('Salmonella AMR MIC Value Prediction Accuracy')

    plt.savefig(snakemake.output[0], dpi=1200)


if __name__ == "__main__":
    main()
