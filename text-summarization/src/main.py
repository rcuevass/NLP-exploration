from utilities.text_extractor import extract_text_from_wikipedia_link
from utilities.summarizer import run_article_summary

if __name__ == '__main__':
    wiki_link = 'https://en.wikipedia.org/wiki/20th_century'
    # wiki_link = 'https://en.wikipedia.org/wiki/Investment_banking'
    article_content = extract_text_from_wikipedia_link(wiki_link)
    print(article_content)
    print('======================================================================')
    summary_results = run_article_summary(article_content)
    print(summary_results)

