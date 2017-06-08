#!/usr/bin/env python3
#  -*- coding: UTF-8 -*-
"""
Poem classifier
"""
import argparse
import glob
import logging
import pprint
import re
import csv
import gc
import pandas
from lxml import etree
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.externals import joblib
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import normalize, Normalizer
from poem_reader import read_xml_directory, parse_text_lines, block_xpath

logging.basicConfig(filename='classifier.log',
                    filemode='a',
                    level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

log = logging.getLogger(__name__)


def read_training_data(path='./'):
    """
    :param path:
    :return: tuple with poem textblocks and other textblocks
    """
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


class TextStats(BaseEstimator, TransformerMixin):
    """Extract features from each document for DictVectorizer"""

    def fit(self, x, y=None):
        return self

    def transform(self, texts):
        capital_regex = re.compile(r'^\s*[A-Z]', re.MULTILINE)
        one_word_regex = re.compile(r'^\S+$', re.MULTILINE)
        stats = [{'row_length': np.average([len(row) for row in text.split('\n')]),
                  'one_word_rows': len(re.findall(one_word_regex, text)),
                  'dots': text.count('.'),
                  'row_start_capitals': len(re.findall(capital_regex, text))
                  }
                 for text in texts]
        # pprint.pprint(stats)
        return stats


def train(poems, nonpoems, quick=False):
    """
    Train the model based on given training data
    :return:
    """
    #nonpoems = nonpoems[::1]

    print(len(poems))
    print(len(nonpoems))

    all_train_data = poems + nonpoems
    all_train_target = [1] * len(poems) + [0] * len(nonpoems)

    all_train_data = [textdata.replace('w', 'v').replace('W', 'V') for textdata in all_train_data]

    tfidf = Pipeline([('vect', CountVectorizer(max_df=1.0, max_features=25400)),
                      ('tfidf', TfidfTransformer())])

    text_feats = Pipeline([('stats', TextStats()),  # returns a list of dicts
                           ('vect', DictVectorizer()),  # list of dicts -> feature matrix
                           ('norm', Normalizer(norm='l2')),
                           ])

    combined_feats = FeatureUnion([('text_feats', text_feats),
                                   ('word_freq', tfidf),
                                   ])

    sgd = SGDClassifier(loss='hinge',
                        penalty='l2',
                        alpha=0.0001,
                        n_iter=6,
                        random_state=42)

    combined_clf = Pipeline([('features', combined_feats),
                             ('clf', sgd),
                             ])

    if quick:
        gs_clf = GridSearchCV(combined_clf, {})
    else:
        parameters = {
            # 'features__word_freq__vect__ngram_range': [(1, 1), (1, 2), (1, 3)],
            'features__word_freq__vect__max_df': [1.0, 0.5],
            'features__word_freq__vect__max_features': [None, 20000, 25000, 25200, 25400, 25600, 26000],
            'features__text_feats__norm__norm': ('l1', 'l2', 'max'),
            'clf__alpha': (1e-3, 1e-4, 1e-5, 1e-6),
            'clf__penalty': ('l2', 'elasticnet'),
            'clf__loss': ('hinge', 'log'),
            'clf__n_iter': (4, 5, 6, 7, 8),
        }

        gs_clf = GridSearchCV(combined_clf, parameters, n_jobs=-1)

    gs_clf.fit(all_train_data, all_train_target)

    predicted = gs_clf.predict(all_train_data)

    print(np.average(predicted))

    print('Final params: %s' % gs_clf.best_params_)
    print('Best score: %s' % gs_clf.best_score_)

    stop_words = gs_clf.best_estimator_.get_params()['features'].get_params().get('word_freq').named_steps['vect'].stop_words_
    print('Number of generated stopwords: %s' % len(stop_words))

    with open('generated_stopwords.txt', 'w', newline='') as fp:
        fp.write('\n'.join(sorted(stop_words)))

    print('Weights %s' % gs_clf.best_estimator_.named_steps['clf'].coef_[0][:4])

    return gs_clf


def parse_metadata_from_path(path):
    """
    Parse metadata from file path, e.g.
    ~/dhh17data/newspapers/newspapers/fin/1820/1457-4888/1457-4888_1820-01-08_1/alto/1457-4888_1820-01-08_1_003.xml
    """

    re_name_split = r'.*/newspapers/newspapers/fin/(.{4})/(.{7,9})/.{7,9}\_.{4}\-(..)\-(..)'

    split = re.search(re_name_split, path)

    year, issn, month, day = split.groups()
    return year, month, day, issn


def get_paper_name_by_issn(issue_df, issn):
    """
    Return paper name, based on its' ISSN number
    """
    try:
        paper = issue_df.loc[issue_df['issn'] == issn]['paper'].iloc[0]
        return paper
    except IndexError:
        log.error('ISSN Number not found: %s' % issn)
        return False


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Textblock classifier to poems and other text")
    argparser.add_argument("--dir", help="Directory to classify")
    argparser.add_argument("--newfile", help="Create new CSV file", dest='newfile', action='store_true')
    argparser.add_argument("--quick", help="Train the model more quick", dest='quick', action='store_true')
    args = argparser.parse_args()

    poems, nonpoems = read_training_data()
    joblib.dump(train(poems, nonpoems, args.quick), 'svm.pkl')

