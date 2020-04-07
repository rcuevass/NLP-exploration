import sys
import re, numpy as np, pandas as pd
from utils.logger import get_log_object
from utils.data_processing import sent_to_words
from pprint import pprint

# Gensim
import gensim, logging, warnings
import gensim.corpora as corpora
from gensim.utils import lemmatize, simple_preprocess
from gensim.models import CoherenceModel
import matplotlib.pyplot as plt

# NLTK Stop words

import nltk
#nltk.download('stopwords')

from nltk.corpus import stopwords

#nltk.download('stopwords')

log = get_log_object()

log.info('Getting stopwords...')
stop_words = stopwords.words('english')
log.info('Stopwords obtained...')
stop_words.extend(['from', 'subject', 're', 'edu', 'use', 'not', 'would', 'say', 'could', '_', 'be', 'know', 'good', 'go', 'get', 'do', 'done', 'try', 'many', 'some', 'nice', 'thank', 'think', 'see', 'rather', 'easy', 'easily', 'lot', 'lack', 'make', 'want', 'seem', 'run', 'need', 'even', 'right', 'line', 'even', 'also', 'may', 'take', 'come'])
log.info('Stopwords extended...')

'''
# Import Dataset
df = pd.read_json('https://raw.githubusercontent.com/selva86/datasets/master/newsgroups.json')
log.info('dataframe obtained...')
df.to_csv('../data/input/news_data_raw.csv')
log.info('saving dataframe to input data')
df = df.loc[df.target_names.isin(['soc.religion.christian', 'rec.sport.hockey', 'talk.politics.mideast', 'rec.motorcycles']) , :]
df.to_csv('../data/output/news_data_preprocessed.csv')
log.info('saving dataframe to output data')
'''

log.info('Reading data')
df = pd.read_csv('../data/output/news_data_preprocessed.csv')

log.info('Converting to list')
data = df.content.values.tolist()
data_words = list(sent_to_words(data))
log.info(str(data_words[:1]))




