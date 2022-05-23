import pandas as pd
import numpy as np
from tqdm.notebook import tqdm


from os.path import isfile, join
from os import listdir

import itertools
from collections import Counter

import re
from setuptools.namespaces import flatten
from urllib.parse import urlparse, unquote_plus


DATA_PATH = '../data/'

#### Define function used

def preprocess_url(url):
    ## convert to urlparse with quoted
    url_parsed = urlparse(unquote_plus(url))
    ## join all url attributes
    url_text = ''.join(x for x in [url_parsed.netloc, url_parsed.path, url_parsed.params, url_parsed.query])
    ## split url to tokens ie: words
    tokens = re.split('[- _ % : , / \. \+ = ]', url_text)
    ## spliting by upper case
    tokens = list(flatten([re.split(r'(?<![A-Z\W])(?=[A-Z])', s) for s in tokens]))
    ## delete token with digits with len < 2
    tokens = [token for token in tokens if (not any(c.isdigit() for c in token)) and (not len(token) <=2)]
    tokens = [token for token in tokens if token not in ['www', 'html', 'com', 'net', 'org']]
    return ' '.join(token for token in tokens)

def keep_selected_labels(labels, labels_to_keep):
    return [label for label in labels if label in labels_to_keep]

## get only files parquet files
onlyfiles = [f for f in listdir(DATA_PATH) if isfile(join(DATA_PATH, f)) and 'snappy.parquet' in f]

dfs = [pd.read_parquet(DATA_PATH + file) for file in onlyfiles]

data = pd.concat(dfs, ignore_index=True)

data['url_cleaned'] = data['url'].apply(preprocess_url)

unique_labels = list(set(itertools.chain(*data.target.values)))

label_count = dict(Counter(flatten(data['target'])))

## Keep only labels where count is greater than threshold
## threshold
threshold = 300
labels_to_keep = [k for k, v in label_count.items() if v > threshold]


data["target_cleaned"] = data["target"].apply(lambda obj: keep_selected_labels(obj, labels_to_keep))    

## delete urls with empty labels
data = data[data['target_cleaned'].apply(lambda x: len(x) != 0)]
data.reset_index(drop=True, inplace=True)


import pickle

## store data
with open(DATA_PATH + 'data_cleaned.pickle', 'wb') as handle:
    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)