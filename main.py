# import util
import sklearn.datasets
import sklearn.metrics
import sklearn.cross_validation
import sklearn.svm
import sklearn.naive_bayes
import sklearn.neighbors
from colorama import init
from termcolor import colored
import sys
import os
# import glob
# from sklearn.datasets import fetch_20newsgroups
import re
import pandas as pd
from sklearn.model_selection import train_test_split
from textblob import Word
import numpy as np
import matplotlib.pyplot as plt
import csv


def main():
    init()
    print '\n\n'

    # load data
    print colored('Loading files into memory', 'green', attrs=['bold'])

    raw_data = pd.read_csv('./datasets/raw_data.csv')
    summ_data = pd.read_csv('./datasets/summ_news.csv')
    summ_title_data = pd.read_csv('./datasets/summ_news_title.csv')
    summ_news_random = pd.read_csv('./datasets/summ_news_random.csv')
    summ_no_title = pd.read_csv('./datasets/summ_no_title.csv')
    print colored('Using Raw Data', 'yellow', attrs=['bold'])
    main_test('Raw_Data',raw_data)
    print colored('Using Summarized Data', 'yellow', attrs=['bold'])
    main_test('Summ_Data',summ_data)
    print colored('Using Summarized Title Data', 'yellow', attrs=['bold'])
    main_test('Summ_Title_Data',summ_title_data)
    print colored('Using Summarized Random Data', 'yellow', attrs=['bold'])
    main_test('Summ_Random_Data',summ_news_random)
    print colored('Using Summarized No Title Data', 'yellow', attrs=['bold'])
    main_test('Summ_No_Title_Data',summ_no_title)



def main_test(d_type,data):


    x = data['news'].tolist()
    y = data['type'].tolist()

    print colored('Processing Data', 'green', attrs=['bold'])
    for index,value in enumerate(x):
        x[index] = ' '.join([Word(word).lemmatize() for word in clean_str(value).split()])

    print colored('Calculating TFIDF', 'green', attrs=['bold'])
    vect = sklearn.feature_extraction.text.TfidfVectorizer(stop_words='english',min_df=2)
    X = vect.fit_transform(x)
    Y = np.array(y)
    Y_target = ['business', 'entertainment', 'politics', 'sport', 'tech']

    print colored('Testing classifier with train-test split', 'magenta', attrs=['bold'])
    arr = ["Naive Bayes","SVM","KNN"]


    for ix in range(2):
        print colored("\n\nUsing " + arr[ix], 'blue', attrs=['bold'])
        clf = get_classifier(ix)

        test_classifier(d_type+"_Using " + arr[ix],X, Y, clf, test_size=0.4, y_names=Y_target, confusion=False)

        count_vector = sklearn.feature_extraction.text.CountVectorizer()
        BOW = count_vector.fit_transform(x)


        print colored('Plotting Data', 'green', attrs=['bold'])
        i, BOW_results = split_test_classifier(clf, BOW, Y)
        i, TF_results = split_test_classifier(clf, X, Y)
        i, TFIDF_results = split_test_classifier(clf, X, Y)

        
        plot_results(d_type+"_Using " + arr[ix],i, [BOW_results, TF_results, TFIDF_results], ['BOW', 'TF', 'TFIDF'])



def get_classifier(ix):
    if ix == 0:
        return sklearn.naive_bayes.MultinomialNB()
    else:
        return sklearn.svm.LinearSVC()
    # else: 
    #     n_neighbors = 11
    #     weights = 'uniform'
    #     weights = 'distance'
    #     return sklearn.neighbors.KNeighborsClassifier(n_neighbors, weights=weights)

def test_classifier(title,X, y, clf, test_size=0.4, y_names=None, confusion=False):
    # train-test split
    print 'test size is: %2.0f%%' % (test_size * 100)
    X_train, X_test, y_train, y_test = sklearn.cross_validation.train_test_split(X, y, test_size=test_size)

    clf.fit(X_train, y_train)
    y_predicted = clf.predict(X_test)

    if not confusion:
        print colored('Classification report', 'magenta', attrs=['bold'])
        report = sklearn.metrics.classification_report(y_test, y_predicted, target_names=y_names)
        classifaction_report_csv(title,report)
    else:
        print colored('Confusion Matrix:', 'magenta', attrs=['bold'])
        print sklearn.metrics.confusion_matrix(y_test, y_predicted)


def split_test_classifier(clf, X, y):
    results = []
    i_ = []
    for i in range(1, 100):
        # print i
        i_.append(i)
        percent = i / 100.0

        # split
        X_train, X_test, y_train, y_test = sklearn.cross_validation.train_test_split(X, y, test_size=percent)

        # learn the model
        clf.fit(X_train, y_train)

        # predict
        y_predicted = clf.predict(X_test)

        # calculate percision
        percision = np.mean(y_predicted == y_test)
        results.append(percision)

    return i_, results

def clean_str(string):
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

def plot_results(title,i, results_list, labels_list):
    colors_list = ['red', 'blue', 'black', 'green', 'cyan', 'yellow']

    plt.figure()
    if not len(results_list) == len(labels_list):
        print 'un equal len in results and labels'
        raise Exception

    for (result, label, color) in zip(results_list, labels_list, colors_list):
        plt.plot(i, result, color=color, lw=2.0, label=label)

    plt.title(title)
    plt.legend()
    # plt.show()
    plt.savefig('Plots/'+title+'.png')

def classifaction_report_csv(title,report):
    report_data = []
    lines = [x for x in report.split('\n') if len(x) > 0]
    for i,line in enumerate(lines[1:]):
        # print line
        row = {}
        row_data = [x for x in line.split("  ") if len(x) > 0]
        row['class'] = row_data[0]
        row['precision'] = float(row_data[1])
        row['recall'] = float(row_data[2])
        row['f1_score'] = float(row_data[3])
        row['support'] = float(row_data[4])
        if i == (len(lines)-2):
            report_data.append({})
        report_data.append(row)
    dataframe = pd.DataFrame.from_dict(report_data)
    dataframe.to_csv('Stats/'+title+'.csv', index = False)

if __name__ == '__main__':
    main()


