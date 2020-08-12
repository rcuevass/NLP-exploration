from utilities.text_extractor import extract_text_from_wikipedia_link
from utilities.summarizer import run_article_summary
from utilities.logger import get_log_object
from utilities.summarizer import min_threshold_summary_length

# instantiate log object
log = get_log_object()


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

