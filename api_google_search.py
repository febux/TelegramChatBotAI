import requests
from bs4 import BeautifulSoup

headers_Get = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/115.0',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US;q=0.5,en;q=0.3',
        'Accept-Encoding': 'gzip, deflate',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Host': 'www.google.com',
        'Alt-Used': 'www.google.com',
    }


def google(q):
    with requests.Session() as s:
        q = '+'.join(q.split())
        url = 'https://www.google.com/search?q=' + q + '&ie=utf-8&oe=utf-8'
        r = s.get(url, headers=headers_Get)

        soup = BeautifulSoup(r.text, "html.parser")
        output = []
        for searchWrapper in soup.find_all('div', {'class': 'MjjYud'}):
            if link := searchWrapper.find('a'):
                url = link.get("href")
                text = link.text.strip()
                result = {'text': text, 'url': url}
            else:
                result = {'text': None}
            output.append(result)

    return output
