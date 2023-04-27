import requests
from bs4 import BeautifulSoup

url = 'https://www.imdb.com/chart/top'

response = requests.get(url)

soup = BeautifulSoup(response.text, 'html.parser')

movies = soup.select('td.titleColumn')

for movie in movies:
    title = movie.select('a')[0].text
    rating_tag = movie.select_one('td.ratingColumn strong') or movie.select_one('td.ratingColumn div.seen-widget')
    rating = rating_tag.text.strip() if rating_tag else 'N/A'
    print(title, rating)

