from utilities.text_extractor import extract_text_from_wikipedia_link
from utilities.summarizer import run_article_summary
import numpy as np
from scipy.stats import mode


def min_threshold(text_to_summarize, min_threshold: int = 5, max_threshold: int = 50) -> int:
    list_thresholds = list(range(min_threshold, max_threshold+1))
    dict_thresholds = dict()
    for thresh_ in list_thresholds:
        text_summarized = run_article_summary(text_to_summarize, substring_threshold_value=thresh_)
        text_summarized_split = text_summarized.split()
        dict_thresholds[thresh_] = len(text_summarized_split)

    np_values = np.array(list(dict_thresholds.values()))
    mode_value = mode(np_values)[0]
    print('Mode ... ', mode_value)

    list_keys = []
    for key_ in dict_thresholds.keys():
        if dict_thresholds[key_] == mode_value:
            list_keys.append(key_)

    print('List = ', list_keys)
    print('min= ', min(list_keys))

    return min(list_keys)


if __name__ == '__main__':
    # wiki_link = 'https://en.wikipedia.org/wiki/20th_century'
    wiki_link = 'https://en.wikipedia.org/wiki/Investment_banking'

    article_content = extract_text_from_wikipedia_link(wiki_link)
    # dict of length for various thresholds
    min_threshold_value = min_threshold(article_content)
    print('============================== Original text ==========================================================')
    print(' ')
    print(article_content)
    print(' ')

    summary_results = run_article_summary(article_content, substring_threshold_value=min_threshold_value,
                                          print_reduction_ratio=True)

    print('============================== Summarized text ==========================================================')
    print(' ')
    print(summary_results)
    print(' ')

