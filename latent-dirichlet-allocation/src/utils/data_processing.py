import re
import pandas as pd
import gensim
from utils.logger import get_log_object
from gensim.utils import simple_preprocess
import spacy

log = get_log_object()


def extract_data():
    df = pd.read_json('https://raw.githubusercontent.com/selva86/datasets/master/newsgroups.json')
    log.info('data frame obtained...')
    df.to_csv('../data/input/news_data_raw.csv')
    log.info('saving data frame to input data')
    df = df.loc[df.target_names.isin(['soc.religion.christian', 'rec.sport.hockey', 'talk.politics.mideast',
                                      'rec.motorcycles']), :]
    df.to_csv('../data/output/news_data_preprocessed.csv')
    log.info('saving data frame to output data')


def sent_to_words(sentences: list) -> list:
    for sent in sentences:
        sent = re.sub('\S*@\S*\s?', '', sent)  # remove emails
        sent = re.sub('\s+', ' ', sent)  # remove newline chars
        sent = re.sub("\'", "", sent)  # remove single quotes
        sent = gensim.utils.simple_preprocess(str(sent), deacc=True)
        yield sent


def get_data_as_list(data_frame) -> list:
    data = data_frame.content.values.tolist()
    data_words = list(sent_to_words(data))
    ##     return data_words[:1]
    return data_words


def get_biagram_triagram_models(data_words: list) -> dict:
    # Build the bigram and trigram models
    bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100)  # higher threshold fewer phrases.
    trigram = gensim.models.Phrases(bigram[data_words], threshold=100)
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)

    dictionary_models = dict({'bigram': bigram_mod, 'trigram': trigram_mod})

    return dictionary_models


def process_words(list_words: list, stop_words: list,
                  allowed_postags) -> list:
    """Remove Stopwords, Form Bigrams, Trigrams and Lemmatization"""
    log.info('Getting N-grams')
    dict_n_grams = get_biagram_triagram_models(list_words)
    bigram_mod = dict_n_grams['bigram']
    trigram_mod = dict_n_grams['trigram']

    log.info('creating texts list')
    texts = [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in list_words]
    texts = [bigram_mod[doc] for doc in texts]
    texts = [trigram_mod[bigram_mod[doc]] for doc in texts]
    texts_out = []
    # load spacy model
    # nlp = spacy.load('en', disable=['parser', 'ner'])
    log.info('loading spaCy model')
    nlp = spacy.load('en_core_web_sm')
    log.info('looping over list of texts...')
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    texts_out = [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts_out]
    return texts_out


