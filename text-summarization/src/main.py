from utilities.text_extractor import extract_text_from_wikipedia_link
from utilities.summarizer import run_article_summary
import numpy as np
from scipy.stats import mode
from utilities.logger import get_log_object

# instantiate log object
log = get_log_object()


def min_threshold_summary_length(text_to_summarize: str, min_threshold: int = 5, max_threshold: int = 50) -> int:
    """
    Function that finds the minimum threshold that provides the same length of summary.
    :param text_to_summarize: string that captures the content of the Wikipedia page
    :param min_threshold: integer -  minimum value to be considered for the consideration of the note length
    :param max_threshold: integer - maximum value to be considered for the consideration of the note length
    :return: minimum value of threshold that provides the same length of summary
    """

    # from min and max values; get a list
    list_thresholds = list(range(min_threshold, max_threshold+1))
    # initialize dictionary
    dict_thresholds = dict()
    # loop over values of threshold given from the input thresholds
    for thresh_ in list_thresholds:
        # obtain summary for the value of threshold in turn
        text_summarized = run_article_summary(text_to_summarize, substring_threshold_value=thresh_)
        # split summary obtained...
        text_summarized_split = text_summarized.split()
        # update dictionary with the corresponding length of summary
        dict_thresholds[thresh_] = len(text_summarized_split)

    # turn list of values from dictionary into numpy array
    np_values = np.array(list(dict_thresholds.values()))
    # capture mode - proxy to most stable length of summary
    mode_value = mode(np_values)[0]

    log.info('Mode for length of summary %f', mode_value)

    # initialize list of keys
    list_keys = []
    # for each key in the dictionary...
    for key_ in dict_thresholds.keys():
        # ... append if value of key in turn matches the mode value
        if dict_thresholds[key_] == mode_value:
            list_keys.append(key_)

    log.info('Minimum threshold value %f ', min(list_keys))

    # give as output the minimum value that provides the same length of summary - mode
    return min(list_keys)


if __name__ == '__main__':
    # link to wikipedia page
    wiki_link = 'https://en.wikipedia.org/wiki/Investment_banking'
    log.info('Wikipedia page to be summarized %s', wiki_link )

    # obtain content of Wiki page associated with link
    article_content = extract_text_from_wikipedia_link(wiki_link)
    # dict of length for various thresholds
    min_threshold_value = min_threshold_summary_length(article_content)
    log.info('============================== Original text =========================================================')
    log.info(' ')
    log.info(article_content)
    log.info(' ')

    # once the minimum threshold has been obtained; generate summary for such threshold
    summary_results = run_article_summary(article_content, substring_threshold_value=min_threshold_value,
                                          print_reduction_ratio=True)

    log.info('============================== Summarized text =======================================================')
    log.info(' ')
    log.info(summary_results)
    log.info(' ')

    # write original and summazarized text to file
    with open("..\\text\\original_article.txt", 'w') as outfile:
        outfile.writelines(article_content)
        outfile.close()

    with open("..\\text\\summarized_article.txt", 'w') as outfile:
        outfile.writelines(summary_results)
        outfile.close()

