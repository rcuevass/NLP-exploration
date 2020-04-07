import sys
import re, numpy as np, pandas as pd
from utils.logger import get_log_object
from utils.data_processing import get_data_as_list, get_biagram_triagram_models, process_words
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

'''
log.info('Extracting data')
extract_data()
log.info('Data extracted')
'''

log.info('Reading data')
df = pd.read_csv('../data/output/news_data_preprocessed.csv')

log.info('Converting to list')
data_words = get_data_as_list(data_frame=df)
log.info(str(data_words[:1]))

#log.info('Get n-grams')
#dictionary_n_grams = get_biagram_triagram_models(data_words=data_words)

log.info('getting processed words')
data_processed = process_words(list_words=data_words,stop_words=stop_words,
                               allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])