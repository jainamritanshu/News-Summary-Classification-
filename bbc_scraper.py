from bs4 import BeautifulSoup
import requests

raw_html = requests.get("http://www.bbc.com/news/entertainment_and_arts").text
parsed_html = BeautifulSoup(raw_html, 'html.parser')
all_e_links = raw_html.find_all("a", "title-link")

enter_news = []

for link in all_e_links:
	link = link['href']
	if not link[:3] == 'http':
		link = "http://bbc.co.uk"+link
		tmp_html = requests.get(link).text
		title = tmp_html.find("h1", "story_body__h1")
		if title is not None:
			content_class = tmp_html.find("div", "story-body_inner")
			content_list = content_class.find_all("p")
			content = "".join(content_list)
			enter_news.append({'title': title, 'content': content})

bus_news = []

raw_html = requests.get("http://www.bbc.com/news/business").text
parsed_html = BeautifulSoup(raw_html, 'html.parser')
all_e_links = raw_html.find_all("a", "title_link__title")
enter_news = []
for link in all_e_links:
	link = link['href']
	if not link[:3] == 'http':
		tmp_html = requests.get(link).text
		title = tmp_html.find("h1", "story_body__h1")
		if title is not None:
			content_class = tmp_html.find("div", "story-body_inner")
			content_list = content_class.find_all("p")
			content = "".join(content_list)
			bus_news.append({'title': title, 'content': content})

tech_news = []

raw_html = requests.get("http://www.bbc.com/news/technology").text
parsed_html = BeautifulSoup(raw_html, 'html.parser')
all_e_links = raw_html.find_all("a", "title_link__title")
enter_news = []
for link in all_e_links:
	link = link['href']
	if not link[:3] == 'http':
		tmp_html = requests.get(link).text
		title = tmp_html.find("h1", "story_body__h1")
		if title is not None:
			content_class = tmp_html.find("div", "story-body_inner")
			content_list = content_class.find_all("p")
			content = "".join(content_list)
			tech_news.append({'title': title, 'content': content})

sport_news = []

raw_html = requests.get("http://www.bbc.com/sport").text
parsed_html = BeautifulSoup(raw_html, 'html.parser')
all_e_links = raw_html.find_all("a", "title_link__title")
enter_news = []
for link in all_e_links:
	link = link['href']
	if not link[:3] == 'http':
		tmp_html = requests.get(link).text
		title = tmp_html.find("h1", "story_body__h1")
		if title is not None:
			content_class = tmp_html.find("div", "story-body_inner")
			content_list = content_class.find_all("p")
			content = "".join(content_list)
			sport_news.append({'title': title, 'content': content})
