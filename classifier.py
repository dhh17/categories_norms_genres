#!/usr/bin/env python3
#  -*- coding: UTF-8 -*-
"""
Poem classifier
"""
import argparse
import glob

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

argparser = argparse.ArgumentParser(description="Textblock classifier to poems and other text")
argparser.add_argument("job", help="Job to do", choices=['train', 'classify'])
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

if args.job == 'train':
    poems, nonpoems = read_training_data()
    print(len(poems))
    print(len(nonpoems))

    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(poems + nonpoems)

    #print(X_train_counts)

    #cc = count_vect.vocabulary_.get('on')
    #print(count_vect.get_feature_names()[cc])
    #print(np.sum(X_train_counts, axis=0).tolist()[0][cc])

    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    print(X_train_tfidf.shape)

