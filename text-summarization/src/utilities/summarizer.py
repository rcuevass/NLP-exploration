# importing libraries
# import nltk
# nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
from utilities.logger import get_log_object

# instantiate log object
log = get_log_object()


def create_dictionary_frequency(text_string: str) -> dict:
    """
    Function that returns frequency of words as a dictionary
    :param text_string: string for which the frequency of terms will be obtained
    :return: frequency_dictionary: dictionary containing the frequency of terms
    """
    # get stop words as a set
    stop_words = set(stopwords.words("english"))

    # tokenize the input string
    words = word_tokenize(text_string)

    # reducing words to their root form
    stem = PorterStemmer()

    # creating dictionary for the word frequency table
    frequency_dictionary = dict()
    for wd in words:
        wd = stem.stem(wd)
        if wd in stop_words:
            continue
        if wd in frequency_dictionary:
            frequency_dictionary[wd] += 1
        else:
            frequency_dictionary[wd] = 1

    return frequency_dictionary


def calculate_sentence_scores(sentences, frequency_table, substring_threshold) -> dict:
    # algorithm for scoring a sentence by its words
    sentence_weight = dict()

    for sentence in sentences:
        # reduced sentence
        sentence_reduced = sentence[:substring_threshold]
        sentence_wordcount_without_stop_words = 0
        for word_weight in frequency_table:
            if word_weight in sentence.lower():
                sentence_wordcount_without_stop_words += 1
                if sentence_reduced in sentence_weight:
                    sentence_weight[sentence_reduced] += frequency_table[word_weight]
                else:
                    sentence_weight[sentence_reduced] = frequency_table[word_weight]

        sentence_weight[sentence_reduced] = sentence_weight[sentence_reduced] / sentence_wordcount_without_stop_words

    return sentence_weight


def calculate_average_score(sentence_weight) -> float:
    # calculating the average score for the sentences
    sum_values = 0
    for entry in sentence_weight:
        sum_values += sentence_weight[entry]

    # getting sentence average value from source text
    average_score = (sum_values / len(sentence_weight))

    return average_score


def get_article_summary(sentences, sentence_weight, threshold, substring_value):
    sentence_counter = 0
    article_summary = ''

    for sentence in sentences:
        sentence_reduced = sentence[:substring_value]
        if sentence_reduced in sentence_weight and sentence_weight[sentence_reduced] >= threshold:
            article_summary += " " + sentence
            sentence_counter += 1

    return article_summary


def run_article_summary(article: str, substring_threshold_value: int = 7, print_reduction_ratio: bool = False) -> str:
    # creating a dictionary for the word frequency table
    frequency_table = create_dictionary_frequency(article)

    # tokenizing the sentences
    sentences = sent_tokenize(article)

    # algorithm for scoring a sentence by its words
    sentence_scores = calculate_sentence_scores(sentences, frequency_table,
                                                substring_threshold=substring_threshold_value)

    # getting the threshold
    threshold = calculate_average_score(sentence_scores)

    # producing the summary
    article_summary = get_article_summary(sentences, sentence_scores, 1.5 * threshold,
                                          substring_value=substring_threshold_value)

    if print_reduction_ratio:
        # length of original document
        initial_length = len(article.split())
        # length of summary
        summary_length = len(article_summary.split())

        log.info('Length of initial document = %i', initial_length)
        log.info('Length of summarized document = %i ', summary_length)

        # reduction
        fraction_reduction = (1-(summary_length/initial_length))*100
        fraction_reduction = round(fraction_reduction, 2)

        log.info('Original text got reduced by %f percent', fraction_reduction)

    return article_summary
