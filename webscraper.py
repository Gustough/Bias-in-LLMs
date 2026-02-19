import bs4
import requests
import sys
import re
import os.path

def get_url(page_number):
    endpoint = f"https://ec.europa.eu/commission/presscorner/api/latestnews?language=en&datefrom=01011978&dateto=31122026&pagesize=20&pagenumber={page_number}"
    r = requests.get(endpoint)
    data = r.json()
    url = data.get("docuLanguageListResources", {})[page_number].get("refCode")
    return url

def scrape_article(url, language):    
    base = 'https://ec.europa.eu/commission/presscorner/api/documents?reference='
    api_url = base + url + f'&language={language}'
    r = requests.get(api_url)
    data = r.json()
    date = data.get("eventDate")
    topics = [item.get("code") for item in data.get("policiesResource", [])]
    html_content = data.get("docuLanguageResource", {}).get("htmlContent", "")
    soup = bs4.BeautifulSoup(html_content, 'html.parser')
    paragraphs = [p.get_text(strip=True) for p in soup.find_all('p')]
    return date, topics, paragraphs


if __name__ == '__main__':
    data_path = "Data"
    language_codes = [
        "bg",  # български
        "cs",  # čeština
        "da",  # dansk
        "de",  # Deutsch
        "el",  # Ελληνικά
        "en",  # English
        "es",  # español
        "et",  # eesti
        "fi",  # suomi
        "fr",  # français
        "ga",  # Gaeilge
        "hr",  # hrvatski
        "hu",  # Magyar
        "it",  # Italiano
        "lt",  # lietuvių kalba
        "lv",  # latviešu valoda
        "mt",  # Malti
        "nl",  # Nederlands
        "pl",  # polski
        "pt",  # Português
        "ro",  # română
        "sk",  # slovenčina
        "sl",  # slovenščina
        "sv",  # svenska
    ]

    for number in range(2, 50):
        for language in language_codes:
            url = get_url(number)
            date, topics, paragraphs = scrape_article(url, language)
            filename = f'{language}-{date}-{number}.txt'
            
            # if not os.path.exists(os.path.join(data_path, filename)):
            #     break
            
            with open(os.path.join(data_path, filename), 'w', encoding="utf-8") as f:
                print(topics, file=f)
                for paragraph in paragraphs:
                    print(paragraph, file=f)
