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

import pandas
from lxml import etree

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
argparser.add_argument("--newfile", help="Create new CSV file", dest='newfile', action='store_true')
args = argparser.parse_args()

logging.basicConfig(filename='classifier.log',
                    filemode='a',
                    level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

log = logging.getLogger(__name__)


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

    nonpoems = nonpoems[::10]  # TODO: Use skipped as test data

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

    test_data = all_train_data[::2]  # TODO: Use less test data
    test_target = all_train_target[::2]

    train_data = all_train_data[1::2]
    train_target = all_train_target[1::2]

    text_clf = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                               alpha=1e-3, n_iter=5, random_state=42)),
                         ])

    text_clf.fit(train_data, train_target)
    predicted = text_clf.predict(test_data)
    acc = np.mean(predicted == test_target)

    print('Cross-validation accuracy %s' % acc)

    parameters = {'vect__ngram_range': [(1, 1), (1, 2), (1, 3)],
                  'tfidf__use_idf': (True, False),
                  'clf__alpha': (1e-2, 1e-3, 1e-4, 1e-5, 1e-6),
                  'clf__penalty': ('l1', 'l2', 'elasticnet'),
                  'clf__loss': ('hinge', 'log'),
                  'clf__n_iter': (3, 4, 5, 6, 7)}

    gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)

    gs_clf = gs_clf.fit(train_data, train_target)

    # _ = text_clf.fit(train_data, train_target)
    predicted = gs_clf.predict(test_data)
    acc = np.mean(predicted == test_target)

    print(predicted)
    print(gs_clf.best_params_)

    print('Parameter-tuned cross-validation accuracy %s' % acc)

    # text_clf.fit(all_train_data, all_train_target)
    gs_clf.fit(all_train_data, all_train_target)

    print('Final params: %s' % gs_clf.best_params_)
    print('Best score: %s' % gs_clf.best_score_)

    return gs_clf


def parse_metadata_from_path(path):
    """
    ~/dhh17data/newspapers/newspapers/fin/1820/1457-4888/1457-4888_1820-01-08_1/alto/1457-4888_1820-01-08_1_003.xml
    """

    re_name_split = r'.*/newspapers/newspapers/fin/(.{4})/(.{7,9})/.{7,9}\_.{4}\-(..)\-(..)'

    split = re.search(re_name_split, path)

    year, issn, month, day = split.groups()
    return year, month, day, issn


if args.job == 'train':
    joblib.dump(train(), 'svm.pkl')

elif args.job == 'predict':
    clf = joblib.load('svm.pkl')

    if args.dir[-1] != '/':
        args.dir = args.dir + '/'

    files = glob.glob(args.dir + "**/*.xml", recursive=True)

    if not files:
        log.warning('No files found for %s' % args.dir)
        quit()

    xmls = []
    for xmlfile in files:
        if 'alto' not in xmlfile:
            continue

        with open(xmlfile, 'r') as f:
            try:
                parsed = etree.parse(f)
                xmls.append((parsed, xmlfile))
                log.debug('Read file %s' % xmlfile)
            except etree.XMLSyntaxError:
                log.error('Error in XML: %s' % xmlfile)

    data = []
    metadata = []
    for xml, filename in xmls:
        text_blocks = block_xpath(xml)

        paper_metadata = parse_metadata_from_path(filename)

        # print(paper_metadata)
        # pprint.pprint(list(parse_text_lines(list(block)) for block in text_blocks))
        # quit()

        for block in text_blocks:
            data.append(parse_text_lines(list(block)))
            metadata.append(paper_metadata + (block.get('ID'),))

    data_orig = data
    data = [d.replace('\n', ' ') for d in data]

    # print(len(data))

    predicted = clf.predict(data)
    #print(predicted)

    # print(metadata[100])
    # print(data_orig[100])
    # quit()

    data_trunc = [d for i, d in enumerate(data_orig) if predicted[i] and len(d) >= 94]
    metadata = [d for i, d in enumerate(metadata) if predicted[i] and len(data_orig[i]) >= 94]

    issues = pandas.read_csv('data/issue_numbers.csv', sep=',')
    with open('foundpoems/found_poems.csv'.format(year=metadata[0][0]), 'w' if args.newfile else 'a', newline='') as fp:
        writer = csv.writer(fp, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        if args.newfile:
            writer.writerow(('Poem', 'Year', 'Month', 'Day', 'Newspaper name', 'ISSN'))

        poemtext = ''
        prev_vector = None
        blockids = []
        for i, d in enumerate(data_trunc):

            # print(metadata[i])

            year, month, day, issn, blockid = metadata[i]

            # if not predicted[i]:
            #     prev_vector = (year, month, day, issn)
            #     continue
            #
            try:
                paper = issues.loc[issues['issn'] == issn]['paper'].iloc[0]
            except IndexError:
                log.error('ISSN Number not found: %s' % issn)

            if prev_vector == (year, month, day, issn):
                if poemtext:
                    poemtext += "\n"
                poemtext += d
                blockids.append(blockid)
            else:
                if poemtext:
                    year2, month2, day2, issn2 = prev_vector
                    paper2 = issues.loc[issues['issn'] == issn2]['paper'].iloc[0]
                    writer.writerow([poemtext.replace('\n', ' '), year2, month2, day2, paper2, issn2])
                    poem_filename = 'foundpoems/{year}_{month}_{day}_{paper} {blocks}'.\
                                    format(year=year2, month=month2, day=day2, paper=paper2, blocks=' '.join(blockids))
                    poem_filename = (poem_filename[:240] + ' TRUNCATED') if len(poem_filename) > 247 else poem_filename
                    poem_filename += '.txt'
                    with open(poem_filename, 'w', newline='') as textp:
                        textp.write(poemtext)

                poemtext = d
                blockids = [blockid]

            prev_vector = (year, month, day, issn)

        writer.writerow([poemtext.replace('\n', ' '), year, month, day, paper, issn])
