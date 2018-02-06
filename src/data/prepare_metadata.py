from shutil import copyfile
import unicodedata
import pandas as pd
import pickle
import numpy as np
import math
from sklearn.externals import joblib

def clean_elem(elem):
    elem = str(elem)
    if '=' in elem:
        new_value = math.log2(float(elem[2:]))
    elif '<' in elem:
        new_value = math.log2(float(elem[1:])/2.0)
    elif '>' in elem:
        new_value = math.log2(float(elem[1:])*2.0)
    else:
        new_value = math.log2(float(elem))
    if np.isnan(new_value): # TODO: Yeah this isn't great
        new_value = 0
    return new_value

def clean_series(series):
    new_series = series.apply(clean_elem)
    return new_series

drugs = ['MIC_AMP', 'MIC_AMC', 'MIC_FOX', 'MIC_CRO', 'MIC_TIO', 'MIC_GEN',
         'MIC_FIS', 'MIC_SXT', 'MIC_AZN', 'MIC_CHL', 'MIC_CIP', 'MIC_NAL',
         'MIC_TET']
convertor = {k: lambda x: unicodedata.normalize("NFKD", x) for k in drugs}

df = pd.read_csv(snakemake.input[0], sep='\t', na_values=['-'], skipfooter=1,
                 skip_blank_lines=False, converters=convertor, engine='python')
df = df.set_index('run')
df = df.filter(regex='MIC')
df = df.rename(columns= lambda x: x.replace('MIC_', ''))

with open(snakemake.output[0], 'wb') as f:
    pickle.dump(df, f)

df = df.apply(clean_series)

with open(snakemake.output[1], 'wb') as f:
    pickle.dump(df, f)

copyfile(snakemake.input[1], snakemake.output[2])
