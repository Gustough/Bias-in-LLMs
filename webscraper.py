import bs4
import requests
import sys
import re
import os.path

def scrape_article(url):    
    ip = url.split('/')[-1].split("_")[1:]
    base = 'https://ec.europa.eu/commission/presscorner/api/documents?reference=IP/'
    
    api_url = base + ip[0] + "/" + ip[1] + '&language=en'
    r = requests.get(api_url)
    data = r.json()
    
    date = data.get("eventDate")
    html_content = data.get("docuLanguageResource", {}).get("htmlContent", "")

    soup = bs4.BeautifulSoup(html_content, 'html.parser')
    
    paragraphs = [p.get_text(strip=True) for p in soup.find_all('p')]
    
    return date, paragraphs


if __name__ == '__main__':
    data_path = "Data"
    for url in sys.argv[1:]:
        date, paragraphs = scrape_article(url)
        if paragraphs:
            for number in range(1, 1000000):
                filename = f'pp-{date}-{number}.txt'
                if not os.path.exists(os.path.join(data_path, filename)):
                    break
            with open(os.path.join(data_path, filename), 'w') as f:
                for paragraph in paragraphs:
                    print(paragraph, file=f)
