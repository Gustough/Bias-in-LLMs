import bs4
import requests
# import sys
# import re
import os.path

def get_url(page_number, id):
    endpoint = f"https://ec.europa.eu/commission/presscorner/api/latestnews?language=en&datefrom=01011978&dateto=31122026&pagesize=20&pagenumber={page_number}"
    r = requests.get(endpoint)
    data = r.json()
    number_of_articles = len(data.get("docuLanguageListResources"))
    url = data.get("docuLanguageListResources", {})[id].get("refCode")
    return url, number_of_articles

def get_languages(url):
    base = 'https://ec.europa.eu/commission/presscorner/api/documents?reference='
    api_url = base + url + "&language=en"
    r = requests.get(api_url)
    data = r.json()
    LANGUAGES = data.get("languages", {})
    languages = list(map(str.lower, LANGUAGES))
    return languages

def scrape_article(url, language):    
    base = 'https://ec.europa.eu/commission/presscorner/api/documents?reference='
    api_url = base + url + f'&language={language}'
    r = requests.get(api_url)
    data = r.json()
    date = data.get("eventDate")
    topics = [item.get("code") for item in data.get("policiesResource", [])]
    topic_descriptions = [item.get("description") for item in data.get("policiesResource", [])]
    html_content = data.get("docuLanguageResource", {}).get("htmlContent", "")
    soup = bs4.BeautifulSoup(html_content, 'html.parser')
    paragraphs = [p.get_text(strip=True) for p in soup.find_all('p')]
    return date, topics, topic_descriptions, paragraphs

if __name__ == '__main__':
    data_path = "Data"
    language_codes = [ # List of languages to scrape
        # "bg",  # български
        # "cs",  # čeština
        # "da",  # dansk
        "de",  # Deutsch
        # "el",  # Ελληνικά
        "en",  # English
        # "es",  # español
        # "et",  # eesti
        # "fi",  # suomi
        # "fr",  # français
        # "ga",  # Gaeilge
        # "hr",  # hrvatski
        # "hu",  # Magyar
        # "it",  # Italiano
        # "lt",  # lietuvių kalba
        # "lv",  # latviešu valoda
        # "mt",  # Malti
        # "nl",  # Nederlands
        # "pl",  # polski
        # "pt",  # Português
        # "ro",  # română
        # "sk",  # slovenčina
        # "sl",  # slovenščina
        # "sv",  # svenska
    ]
    topic_list = set()
    counter = 1
    target_articles = 11 # Number of articles to scrape -1
    
    for page in range(1, 1000):
        if counter >= target_articles:
            break

        _, article_ids = get_url(counter, 0)
        
        for article_id in range(article_ids):
            if counter >= target_articles:
                break
            
            url, _ = get_url(counter, article_id)
            languages = get_languages(url)
            languages.append("en")
            
            if not all(code in languages for code in language_codes):
                continue
            
            for language in languages:
                if language in language_codes:
                    date, topics, topic_descriptions, paragraphs = scrape_article(url, language)
                    filename = f'{counter}-{language}-{date}.txt'
                    
                    with open(os.path.join(data_path, filename), 'w', encoding="utf-8") as f:
                        print(url.split("/")[0], file=f)
                        print(topics, file=f)
                        for paragraph in paragraphs:
                            print(paragraph, file=f)         
            
            for topic, description in zip(topics, topic_descriptions):
                with open(os.path.join(data_path, "topic_map.txt"), 'r', encoding="utf-8") as g:
                    lines = g.readlines()
                    for line in lines:
                        topic_list.add(line.split(":")[0])
                    if topic not in topic_list and language == "en":        
                        with open(os.path.join(data_path, "topic_map.txt"), 'a', encoding="utf-8") as g:
                            print(f"{topic}: {description}", file=g)

            counter += 1    
            