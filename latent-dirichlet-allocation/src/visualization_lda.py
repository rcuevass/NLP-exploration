import sys
import re, numpy as np, pandas as pd

from utils.data_processing import get_data_as_list, process_words
from pprint import pprint
from utils.logger import get_log_object

# Gensim
import gensim, logging, warnings
import gensim.corpora as corpora
from gensim.utils import lemmatize, simple_preprocess
from gensim.models import CoherenceModel
import pyLDAvis.gensim
import pickle


log = get_log_object()

# Load corpus from pkl file
pkl_filename = '../corpus/corpus.pkl'
with open(pkl_filename, 'rb') as file:
    corpus = pickle.load(file)

# Load model from pkl file
pkl_filename = '../models/lda_model.pkl'
with open(pkl_filename, 'rb') as file:
    lda_model = pickle.load(file)

log.info('topics...')
log.info(str(pprint(lda_model.print_topics())))

vis = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary=lda_model.id2word)

'''
# to review: https://stackoverflow.com/questions/43317056/pyldavis-unable-to-view-the-graph
#pyLDAvis.show(data=vis)

'''