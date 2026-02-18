import bs4
import requests
import sys
import re
import os.path

def scrape_article(url, language):    
    ip = url.split('/')[-1].split("_")[1:]
    base = 'https://ec.europa.eu/commission/presscorner/api/documents?reference=IP/'
    
    api_url = base + ip[0] + "/" + ip[1] + f'&language={language}'
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

    for url in sys.argv[1:]:
        for language in language_codes:
            date, topics, paragraphs = scrape_article(url, language)
            if paragraphs:
                for number in range(1, 1000000):
                    filename = f'{language}-{date}-{number}.txt'
                    if not os.path.exists(os.path.join(data_path, filename)):
                        break
                with open(os.path.join(data_path, filename), 'w', encoding="utf-8") as f:
                    print(topics, file=f)
                    for paragraph in paragraphs:
                        print(paragraph, file=f)
