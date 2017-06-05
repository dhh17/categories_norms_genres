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

    tfidf = Pipeline([('vect', CountVectorizer()),
                      ('tfidf', TfidfTransformer())])

    text_feats = Pipeline([('stats', TextStats()),  # returns a list of dicts
                           ('vect', DictVectorizer()),  # list of dicts -> feature matrix
                           ('norm', Normalizer()),
                           ])

    combined_feats = FeatureUnion([('text_feats', text_feats),
                                   ('word_freq', tfidf),
                                   ])

    sgd = SGDClassifier(loss='hinge',
                        penalty='l2',
                        alpha=0.0001,
                        n_iter=5,
                        random_state=42)

    combined_clf = Pipeline([('features', combined_feats),
                             ('clf', sgd),
                             ])

    if quick:
        test_data = all_train_data[::2]
        test_target = all_train_target[::2]

        train_data = all_train_data[1::2]
        train_target = all_train_target[1::2]

        combined_clf.fit(train_data, train_target)
        predicted = combined_clf.predict(test_data)
        acc = np.mean(predicted == test_target)

        # text_clf = Pipeline([('vect', CountVectorizer()),
        #                      ('tfidf', TfidfTransformer()),
        #                      ('clf', SGDClassifier(loss='hinge', penalty='l2',
        #                                            alpha=1e-3, n_iter=5, random_state=42)),
        #                      ])
        #
        # text_clf.fit(train_data, train_target)
        # predicted = text_clf.predict(test_data)
        # acc = np.mean(predicted == test_target)

        print('Cross-validation accuracy %s' % acc)
        print('Text feature weights %s' % sgd.coef_[0][:4])

        return combined_clf

    parameters = {
        # 'features__word_freq__vect__ngram_range': [(1, 1), (1, 2), (1, 3)],
        'features__word_freq__vect__max_df': [0.9, 0.8, 0.7, 0.6, 0.5],
        'features__word_freq__vect__max_features': [None, 500, 1000, 1500, 10000, 20000],
        'features__text_feats__norm__norm': ('l1', 'l2', 'max'),
        'clf__alpha': (1e-3, 1e-4, 1e-5, 1e-6),
         'clf__penalty': ('l1', 'l2', 'elasticnet'),
        # 'clf__loss': ('hinge', 'log'),
        'clf__n_iter': (3, 4, 5, 6),
    }

    gs_clf = GridSearchCV(combined_clf, parameters, n_jobs=-1)

    # gs_clf = gs_clf.fit(train_data, train_target)
    #
    # print(gs_clf.best_params_)
    # print('Parameter-tuned cross-validation accuracy %s' % acc)

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
    argparser.add_argument("job", help="Job to do", choices=['train', 'predict'])
    argparser.add_argument("--dir", help="Directory to classify")
    argparser.add_argument("--newfile", help="Create new CSV file", dest='newfile', action='store_true')
    argparser.add_argument("--quick", help="Train the model more quick", dest='quick', action='store_true')
    args = argparser.parse_args()

    if args.job == 'train':
        poems, nonpoems = read_training_data()
        joblib.dump(train(poems, nonpoems, args.quick), 'svm.pkl')

    elif args.job == 'predict':
        # THIS PART OF CODE IS RATHER OBSOLETE, USE CLASSIFIER2 INSTEAD TO DO PREDICTIONS

        log.info('Loading classifier from pickle file')
        clf = joblib.load('svm.pkl')

        log.info('Classifier loaded')

        if args.dir[-1] != '/':
            args.dir = args.dir + '/'

        files = glob.glob(args.dir + "**/*.xml", recursive=True)

        if not files:
            log.warning('No files found for %s' % args.dir)
            quit()
        else:
            log.info('Found %s XML files' % len(files))

        xmls = []
        for filename in files:
            if 'alto' not in filename:
                continue

            with open(filename, 'r') as f:
                try:
                    parsed = etree.parse(f)
                    xmls.append((parsed, filename))
                    log.debug('Read file %s' % filename)
                except etree.XMLSyntaxError:
                    log.error('Error in XML: %s' % filename)

        data = []
        metadata = []
        for xml, filename in xmls:
            text_blocks = block_xpath(xml)

            paper_metadata = parse_metadata_from_path(filename)

            for block in text_blocks:
                data.append(parse_text_lines(list(block)))
                metadata.append(paper_metadata + (block.get('ID'),))

        log.info('All XML files have been read')

        data_orig = data
        data = [d.replace('\n', ' ') for d in data]

        log.info('Doing predictions.')

        predicted = clf.predict(data)

        data_trunc = tuple(d for i, d in enumerate(data_orig) if predicted[i] and len(d) >= 94)
        metadata = tuple(d for i, d in enumerate(metadata) if predicted[i] and len(data_orig[i]) >= 94)

        log.info('Predictions done, writing results to files.')

        issues = pandas.read_csv('data/issue_numbers.csv', sep=',')
        with open('foundpoems/found_poems.csv'.format(year=metadata[0][0]), 'w' if args.newfile else 'a',
                  newline='') as fp:
            writer = csv.writer(fp, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

            if args.newfile:
                writer.writerow(('Poem', 'Year', 'Month', 'Day', 'Newspaper name', 'ISSN'))
                log.info('Created new CSV file')

            poemtext = ''
            prev_vector = None
            blockids = []
            for i, d in enumerate(data_trunc):

                year, month, day, issn, blockid = metadata[i]

                paper = get_paper_name_by_issn(issues, issn)

                if prev_vector == (year, month, day, issn):
                    if poemtext:
                        poemtext += "\n"
                    poemtext += d
                    blockids.append(blockid)
                else:
                    if poemtext:
                        year2, month2, day2, issn2 = prev_vector
                        paper2 = get_paper_name_by_issn(issues, issn2)
                        writer.writerow([poemtext.replace('\n', ' '), year2, month2, day2, paper2, issn2])
                        poem_filename = 'foundpoems/{year}_{month}_{day}_{paper} {blocks}'. \
                            format(year=year2, month=month2, day=day2, paper=paper2, blocks=' '.join(blockids))
                        poem_filename = (poem_filename[:240] + ' TRUNCATED') if len(
                            poem_filename) > 247 else poem_filename
                        poem_filename += '.txt'
                        with open(poem_filename, 'w', newline='') as textp:
                            textp.write(poemtext)

                    poemtext = d
                    blockids = [blockid]

                prev_vector = (year, month, day, issn)

            writer.writerow([poemtext.replace('\n', ' '), year, month, day, paper, issn])
            log.info('Updated CSV file for year %s' % metadata[0][0])
