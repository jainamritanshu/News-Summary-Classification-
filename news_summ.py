# -*- coding: utf-8 -*-
from nltk.tokenize import sent_tokenize,word_tokenize
from nltk.corpus import stopwords
from collections import defaultdict
from string import punctuation
from heapq import nlargest
import urllib2
from bs4 import BeautifulSoup
import pandas as pd
import os

BASE_DIR = os.getcwd()

class FrequencySummarizer:
  def __init__(self, min_cut=0.1, max_cut=0.9):
    """
     Initilize the text summarizer.
     Words that have a frequency term lower than min_cut 
     or higer than max_cut will be ignored.
    """
    self._min_cut = min_cut
    self._max_cut = max_cut 
    self._stopwords = set(stopwords.words('english') + list(punctuation))

  def _compute_frequencies(self, word_sent):
    """ 
      Compute the frequency of each of word.
      Input: 
       word_sent, a list of sentences already tokenized.
      Output: 
       freq, a dictionary where freq[w] is the frequency of w.
    """
    freq = defaultdict(int)
    for s in word_sent:
      for word in s:
        if word not in self._stopwords:
          freq[word] += 1
    # frequencies normalization and fitering
    m = float(max(freq.values()))
    for w in freq.keys():
      freq[w] = freq[w]/m
      if freq[w] >= self._max_cut or freq[w] <= self._min_cut:
        del freq[w]
    return freq

  def summarize(self, text, n):
    """
      Return a list of n sentences 
      which represent the summary of text.
    """
    try:
        sents = sent_tokenize(text)
        assert n <= len(sents)
        word_sent = [word_tokenize(s.lower()) for s in sents]
        self._freq = self._compute_frequencies(word_sent)
        ranking = defaultdict(int)
        for i,sent in enumerate(word_sent):
          for w in sent:
            if w in self._freq:
              ranking[i] += self._freq[w]
        sents_idx = self._rank(ranking, n)  
        return [sents[j] for j in sents_idx]
    except:
        return "unicode error"

  def _rank(self, ranking, n):
    """ return the first n sentences with highest ranking """
    return nlargest(n, ranking, key=ranking.get)

def get_only_text(url):
    """ 
      return the title and the text of the article
      at the specified url
    """
    page = urllib2.urlopen(url).read()
    soup = BeautifulSoup(page)
    text = ' '.join(map(lambda p: p.text, soup.find_all('p')))
    return soup.title.text, text

import re
import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import Word
import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix


def clean_str(string):
    """
    Tokenization/string cleaning for datasets.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"\'s", "", string)
    string = re.sub(r"\'ve", "", string)
    string = re.sub(r"n\'t", "", string)
    string = re.sub(r"\'re", "", string)
    string = re.sub(r"\'d", "", string)
    string = re.sub(r"\'ll", "", string)
    string = re.sub(r",", "", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", "", string)
    string = re.sub(r"\)", "", string)
    string = re.sub(r"\?", "", string)
    string = re.sub(r"'", "", string)
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"[0-9]\w+|[0-9]","", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


if __name__ == '__main__':
    # feed_xml = urllib2.urlopen('http://feeds.bbci.co.uk/news/rss.xml').read()
    # feed = BeautifulSoup(feed_xml, "lxml")
    # to_summarize = map(lambda p: p.text, feed.find_all('guid'))

    fs = FrequencySummarizer()

    data = pd.read_csv('./news-dataset.csv')
    x = data['news'].tolist()
    y = data['type'].tolist()

    summary = []
    df_txt = []
    df_type = []
    for index,value in enumerate(x):
        print "processing data:",index
        title = ""
        z = 0
        # print value[0]
        # print value[1]
        one_space = False
        while z<len(value):
            if value[z].isspace() and one_space == True:
                break
            elif value[z].isspace() and one_space == False:    
                one_space = True
            else:
                one_space = False
            # print value[z]
            title+=value[z]
            z+=1
        x[index] = value # ' '.join([Word(word).lemmatize() for word in clean_str(value).split()])
        
    # for article_url in to_summarize[:5]:
        content = x[index][z:]
        title, text = (title, content)
        print '----------------------------------'
        print title
        sum_tmp = []
        tmp_fl = True
        for s in fs.summarize(content, 3):
            if not s is "unicode error":
                sum_tmp.append(s)
            else:
                tmp_fl = False

        if tmp_fl == True:
            sum_tmp.insert(0, title)
            text = "".join(sum_tmp)
            summary.append({'title': title, 'text': text})
            df_txt.append(text)
            df_type.append(y[index])
            print text+"\n"

    processed_lists = list(zip(df_txt, df_type))
    data_frame = pd.DataFrame(data = processed_lists, columns=['news', 'type'])
    data_frame.to_csv(os.path.join(BASE_DIR, 'summ_news.csv'))