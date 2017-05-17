#!/usr/bin/env python3
#  -*- coding: UTF-8 -*-
"""
Poem classifier
"""
import argparse
import glob

import numpy as np
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from poem_reader import read_xml_directory, parse_text_lines, block_xpath

argparser = argparse.ArgumentParser(description="Textblock classifier to poems and other text")
argparser.add_argument("job", help="Job to do", choices=['train', 'predict'])
argparser.add_argument("--dir", help="Directory to classify")
args = argparser.parse_args()


def read_training_data(path='./'):
    poem_files = glob.glob(path + "poemblocks/*.txt")
    nonpoem_files = glob.glob(path + "nonpoemblocks/*.txt")

    poems = []
    for poem_file in poem_files:
        with open(poem_file, 'r') as f:
            poems.append(f.read())

    nonpoems = []
    for nonpoem_file in nonpoem_files:
        with open(nonpoem_file, 'r') as f:
            nonpoems.append(f.read())

    return poems, nonpoems


def train():
    poems, nonpoems = read_training_data()
    print(len(poems))
    print(len(nonpoems))

    # count_vect = CountVectorizer()
    # X_train_counts = count_vect.fit_transform(poems + nonpoems)
    #
    # #print(X_train_counts)
    #
    # #cc = count_vect.vocabulary_.get('on')
    # #print(count_vect.get_feature_names()[cc])
    # #print(np.sum(X_train_counts, axis=0).tolist()[0][cc])
    #
    # tfidf_transformer = TfidfTransformer()
    # X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    # print(X_train_tfidf.shape)

    all_train_data = poems + nonpoems
    all_train_target = [1] * len(poems) + [0] * len(nonpoems)

    all_train_data = [d.replace('\n', ' ') for d in all_train_data]

    test_data = all_train_data[::2]
    test_target = all_train_target[::2]

    train_data = all_train_data[1::2]
    train_target = all_train_target[1::2]

    text_clf = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                               alpha=1e-3, n_iter=5, random_state=42)),
                         ])

    _ = text_clf.fit(train_data, train_target)
    predicted = text_clf.predict(test_data)
    acc = np.mean(predicted == test_target)

    print('Cross-validation accuracy %s' % acc)

    # parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
    #               'tfidf__use_idf': (True, False),
    #               'clf__alpha': (1e-2, 1e-3)}
    #
    # gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)
    #
    # gs_clf = gs_clf.fit(train_data, train_target)
    #
    # # _ = text_clf.fit(train_data, train_target)
    # predicted = gs_clf.predict(test_data)
    # acc = np.mean(predicted == test_target)
    #
    # print('Cross-validation accuracy %s' % acc)

    _ = text_clf.fit(all_train_data, all_train_target)

    return text_clf


if args.job == 'train':
    joblib.dump(train(), 'svm.pkl')

elif args.job == 'predict':
    clf = joblib.load('svm.pkl')
    xmls = read_xml_directory(args.dir)

    data = []
    for xml in xmls:
        text_blocks = block_xpath(xml)

        for block in text_blocks:
            data.append(parse_text_lines(list(block)))

    data = [d.replace('\n', ' ') for d in data]

    print(data[:3])
    print(len(data))

    predicted = clf.predict(data)
    print(predicted)
