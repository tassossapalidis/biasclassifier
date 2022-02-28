import argparse
from newspaper import Article

def get_content(url):
    '''
    Returns the title and content of news article at the specified url
    input: URL (as string)
    output: title of article, article content body (as strings)
    '''

    article = Article(url)
    article.download()
    article.parse()
    
    return article.title.strip().replace('\n',''), article.text.strip().replace('\n','')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'scrape article')
    parser.add_argument('--url', type = str, help = 'url of news article')
    args = parser.parse_args()

    print(get_content(args.url))