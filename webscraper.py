import bs4
import requests
import os.path

def get_url(page_number, id):
    endpoint = f"https://ec.europa.eu/commission/presscorner/api/search?language=en&ts=1773838193030&documentTypeCodes=&pagesize=20&pagenumber={page_number}&datefrom=01031976"
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
    document_type = data.get("docutypeResource", {}).get("code")
    document_description = data.get("docutypeResource", {}).get("description")
    html_content = data.get("docuLanguageResource", {}).get("htmlContent", "")
    soup = bs4.BeautifulSoup(html_content, 'html.parser')
    paragraphs = [p.get_text(strip=True) for p in soup.find_all('p')]
    link_tag = url.replace("/", "_")
    link = f"https://ec.europa.eu/commission/presscorner/detail/en/{link_tag}"
    return date, topics, topic_descriptions, document_type, document_description, paragraphs, link

if __name__ == '__main__':
    data_path = "Data"
    existing_files = os.listdir(data_path)
    existing_indices = []
    existing_links = set()
    
    for f in existing_files:
        try:
            num = int(f.split("-", 1)[0])
            existing_indices.append(num)

            with open(os.path.join(data_path, f), "r", encoding="utf-8") as file:
                lines = file.readlines()
                if len(lines) >= 3:
                    existing_links.add(lines[2].strip())
        except:
            continue

    counter = max(existing_indices, default=0) + 1

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
    document_type_list = set()
    target_articles = 201 # Number of articles to scrape -1
    
    for page in range(1, 10000):
        if counter >= target_articles:
            break

        _, article_ids = get_url(counter, 0)
        
        for article_id in range(article_ids):
            count = False
            if counter >= target_articles:
                break

            url, _ = get_url(page, article_id)
            languages = get_languages(url)
            languages.append("en")
            
            if not all(code in languages for code in language_codes):
                continue
            
            for language in languages:
                if language in language_codes:
                    date, topics, topic_descriptions, d_type, d_desc, paragraphs, link = scrape_article(url, language)
                    filename = f'{counter}-{language}-{date}.txt'
                    
                    with open(os.path.join(data_path, filename), 'w', encoding="utf-8") as f:
                        print(url.split("/")[0], file=f)
                        print(topics, file=f)
                        print(link, "\n", file=f)
                        full_text = "\n\n".join(paragraphs)
                        print(full_text, file=f)
                        count = True
                            
            for topic, description in zip(topics, topic_descriptions):
                with open(os.path.join(data_path, "topic_map.txt"), 'r', encoding="utf-8") as g:
                    topic_list = {line.split(":")[0] for line in g}
                    if topic not in topic_list and language == "en":        
                        with open(os.path.join(data_path, "topic_map.txt"), 'a', encoding="utf-8") as g:
                            print(f"{topic}: {description}", file=g)
                            
            with open(os.path.join(data_path, "document_type_map.txt"), 'r', encoding="utf-8") as h:
                document_type_list = {line.split(":")[0] for line in h}
                if d_type not in document_type_list and language == "en":
                    with open(os.path.join(data_path, "document_type_map.txt"), 'a', encoding="utf-8") as h:
                        print(f"{d_type}: {d_desc}", file=h)

            if count == True: counter += 1