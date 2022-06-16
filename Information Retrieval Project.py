# Requirements
"""

pip install sklearn numpy pandas requests jdatetime bs4 newspaper3k google.colab

"""# Imports"""

import numpy
import pandas
import requests
import jdatetime

from bs4 import BeautifulSoup
from newspaper import Article
from google.colab import drive
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

"""# Crawling"""

collected_data = []
main_url = 'https://www.khabaronline.ir'

def get_html_content(date, page):

  date = date.strftime('%Y-%m-%d')
  page_url = f'{main_url}/page/archive.xhtml?date={date}&pi={page}'

  return requests.get(url= page_url).text

def get_article_links(date, page):
  
  html_content = get_html_content(date= date, page= page)

  html_parser = BeautifulSoup(markup= html_content, features= 'html.parser')

  html_a_links = html_parser.select(selector= 'ul li.news h3 a')

  article_links = []

  for a_link in html_a_links:

    article_links.append(a_link['href'])

  return article_links

def parse_article_content(url):

  article_url = f'{main_url}{url}'

  article = Article(url= article_url, language= 'fa')

  article.download()

  article.parse()
  
  article_content = {
    'url': article.url,
    'id': article.meta_data['nastooh']['nid'],
    'title': article.title,
    'image': article.top_img,
    'summary': article.meta_description,
    'text': article.text,
    'tags': article.tags,
    'publish': article.publish_date,
    'keywords': article.meta_keywords,
  }

  return article_content

def save_collected_data(current_date, from_date, to_date):

  try:

    df = pandas.DataFrame(data= collected_data)

    from_date = from_date.strftime("%Y-%m-%d")
    to_date = to_date.strftime("%Y-%m-%d")

    data_file_name = f'khabaronline-{from_date}-{to_date}.csv'

    df.to_csv(data_file_name, mode='a', index=False, header=False)

    print(f'\nAll colleted data in date: {current_date} has been saved.\n')

  except:
    print(f'\nSomething went wrong when trying to save colleted data in date: {current_date}.\n')

def crawl_khabaronline(from_date, to_date):
  
  page = 0
  current_date = from_date

  while True:

    try:

      page += 1
      print(f'\nCollecting articles in date: {current_date.strftime("%Y-%m-%d")} and page: {page}:')
      article_links = get_article_links(date= current_date, page= page)
      
      if len(article_links) != 0:

        for article_url in article_links:

          try:
            article_content = parse_article_content(url= article_url)
            print(f'\tParsed article with id: {article_content["id"]}')
            collected_data.append(article_content)
          except:
            print(f'Something went wrong when trying to parse article with (article_url:{article_url}) parameters.')        
            continue

      else:
        if current_date == to_date:
          break
        else:
          page = 0
          save_collected_data(current_date= current_date, from_date= from_date, to_date= to_date)
          current_date = current_date + jdatetime.timedelta(days= 1)
          collected_data.clear()

    except:
      print(f'Something went wrong when trying to get article links with (date:{current_date.strftime("%Y-%m-%d")}, page: {page}) parameters.')    
      continue

# usage

from_date = jdatetime.date(year= 1400, month= 1, day= 1)
to_date = jdatetime.date(year= 1400, month= 4, day= 31)

crawl_khabaronline(from_date= from_date, to_date = to_date)

"""# Reading Collected Data"""

# load data from google drive

drive.mount(mountpoint= '/content/drive')

data_file_path = '/content/drive/MyDrive/Colab Notebooks/khabaronline-1400-01-01-1400-04-31.csv'

documents = pandas.read_csv(filepath_or_buffer= data_file_path)

"""# EDA Collected Data"""

# shape (number of rows and columns)

documents.shape

# column names

documents.columns

# top eight documents

documents.head(8)

"""# Tf-Idf Documents Vectorization"""

vectorizer = TfidfVectorizer()

vectorized_documents = vectorizer.fit_transform(raw_documents= documents['summary'])

"""# EDA Vectorized Documents"""

# shape (number of rows (documents) and columns (vocabulary))

vectorized_documents.shape

# number of tf-idf vocabulary

len(vectorizer.vocabulary_)

# top eight vocabulary words

list(vectorizer.vocabulary_.keys())[:8]

"""# Tf-Idf Query Vectorization"""

query = 'قیمت امروز دلار'

vectorized_query = vectorizer.transform(raw_documents= [query])[0]

"""# Calculating Similarities"""

query_document_similarities = []

for document in vectorized_documents:
  similarity = float(cosine_similarity(document, vectorized_query))
  query_document_similarities.append(similarity)

"""# Sorting Results"""

result_count = 20
sorted_indexes = numpy.argsort(query_document_similarities)

for i in range(result_count):
  current_index = sorted_indexes[-i-1]
  current_document = documents.iloc[current_index]
  print('similarity:', query_document_similarities[current_index], 'title:', current_document['title'], 'url:', current_document['url'])