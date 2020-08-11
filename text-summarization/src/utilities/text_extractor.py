# importing libraries
import bs4 as BeautifulSoup
import urllib.request


def extract_text_from_wikipedia_link(url_wikipedia_link: str) -> str:
    """
    Function that scrapes text associated with a wikipedia page based on its link to it
    :param url_wikipedia_link: string capturing the url of a wikipedia page
    :return: article_content - string capturing the text from the Wikipedia page
    """
    # fetching the content from the URL
    fetched_data = urllib.request.urlopen(url_wikipedia_link)
    article_read = fetched_data.read()
    # parsing the URL content and storing in a variable
    article_parsed = BeautifulSoup.BeautifulSoup(article_read, 'html.parser')
    # returning <p> tags
    paragraphs = article_parsed.find_all('p')

    # Initialize string that will be returned; content of wikipedia article
    article_content = ''

    # looping through the paragraphs and adding them to the variable
    for p in paragraphs:
        article_content += p.text

    # return content
    return article_content
