#!/usr/bin/env python3
#  -*- coding: UTF-8 -*-
"""
Poem classifier, refactored to process single XML file at once
"""
import argparse
import glob
import logging
import re
import csv
from collections import defaultdict
from itertools import chain

import pandas
from lxml import etree
from sklearn.externals import joblib


from classifier_train import parse_metadata_from_path, get_paper_name_by_issn, TextStats
from poem_reader import read_xml_directory, parse_text_lines, block_xpath, read_blocks_from_csv

logging.basicConfig(filename='classifier.log',
                    filemode='a',
                    level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

log = logging.getLogger(__name__)


def classify_xmls(path, newfile):
    files = glob.glob(path + "**/*.xml", recursive=True)

    if not files:
        log.warning('No files found for %s' % path)
        quit()
    else:
        log.info('Found %s XML files' % len(files))

    if newfile:
        with open('foundpoems/found_poems.csv', 'w', newline='') as fp:
            writer = csv.writer(fp, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(('Poem', 'Year', 'Month', 'Day', 'Newspaper name', 'ISSN'))
            log.info('Created new CSV file')

    issues = pandas.read_csv('data/issue_numbers.csv', sep=',')

    filegroup = defaultdict(list)
    filenamesplitter = r'(.*)/[a-zA-Z0-9\-\_]+\.xml'

    for filename in files:
        if 'alto' not in filename:
            continue
        split = re.search(filenamesplitter, filename)
        if split:
            file_prefix = split.groups()
            filegroup[file_prefix].append(filename)

    log.info('Found %s newspapers, with %s files for path %s' % (len(filegroup),
                                                                  len(list(chain(*filegroup.values()))),
                                                                  path))

    for issue, issue_files in filegroup.items():
        data = []
        metadata = []

        # TODO: Order the textblocks based on textblock coordinates

        for filename in issue_files:

            parsed = None
            with open(filename, 'r') as f:
                try:
                    parsed = etree.parse(f)
                    log.debug('Read file %s' % filename)
                except etree.XMLSyntaxError:
                    log.error('Error in XML: %s' % filename)

            if not parsed:
                continue

            text_blocks = block_xpath(parsed)
            paper_metadata = parse_metadata_from_path(filename)

            for block in text_blocks:
                data.append(parse_text_lines(list(block)).replace('w', 'v').replace('W', 'V'))
                metadata.append(paper_metadata + (block.get('ID'),))
                # Metadata format: (year, month, day, issn, blockid)

        if not data:
            continue  # One XML file has no text blocks

        data_orig = data
        # data = [d.replace('\n', ' ') for d in data]

        predicted = clf.predict(data)

        data_trunc = [d for i, d in enumerate(data_orig) if predicted[i] and len(d) >= 94]
        metadata = [d for i, d in enumerate(metadata) if predicted[i] and len(data_orig[i]) >= 94]

        if not data_trunc:
            continue

        blockids = [md[-1] for md in metadata]
        poemtext = '\n'.join(d for d in data_trunc)

        year, month, day, issn, _ = metadata[0]  # Just take first one because they all should be same

        paper = get_paper_name_by_issn(issues, issn) or issn

        with open('foundpoems/found_poems.csv', 'a', newline='') as fp:
            writer = csv.writer(fp, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow([poemtext, year, month, day, paper, issn])

            log.debug('Updated CSV file')

        poem_filename = 'foundpoems/{year}_{month}_{day}_{paper} {blocks}'. \
            format(year=year, month=month, day=day, paper=paper, blocks=' '.join(blockids))
        poem_filename = (poem_filename[:240] + ' TRUNCATED') if len(poem_filename) > 247 else poem_filename
        poem_filename += '.txt'

        with open(poem_filename, 'w', newline='') as textp:
            textp.write(poemtext)
            log.debug('Written poem to file %s' % poem_filename)


def classify_csv(path, newfile):
    """
    Classify text blocks from a CSV file

    :param path: Path to CSV file
    """
    blockgroups_df = pandas.read_csv(path, header=None, sep=",")

    if blockgroups_df.empty:
        log.warning('No rows found for %s' % path)
        quit()
    else:
        log.info('Found %s rows from CSV file' % len(blockgroups_df))

    if newfile:
        with open('foundpoems/csv_found_poems.csv', 'w', newline='') as fp:
            writer = csv.writer(fp, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(('Poem', 'Year', 'Month', 'Day', 'Newspaper name', 'ISSN'))
            log.info('Created new CSV file')

        with open('foundpoems/csv_found_nonpoems.csv', 'w', newline='') as fp:
            writer = csv.writer(fp, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(('Nonpoem text', 'Year', 'Month', 'Day', 'Newspaper name', 'ISSN'))
            log.info('Created new CSV file')

    for blockgroup in blockgroups_df.iterrows():
        textblocks, year, month, day, paper, issn = blockgroup[1]
        textblocks = textblocks.replace('w', 'v').replace('W', 'V')  # Not needed for classifier generated files, but others

        textblocks = textblocks.split('\n\n')

        predicted = clf.predict(textblocks)

        poems = '\n\n'.join(block for block, c in zip(textblocks, predicted) if c)
        nonpoems = '\n\n'.join(block for block, c in zip(textblocks, predicted) if not c)

        if poems:
            with open('foundpoems/csv_found_poems.csv', 'a', newline='') as fp:
                writer = csv.writer(fp, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                writer.writerow([poems, year, month, day, paper, issn])

        if nonpoems:
            with open('foundpoems/csv_found_nonpoems.csv', 'a', newline='') as fp:
                writer = csv.writer(fp, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                writer.writerow([nonpoems, year, month, day, paper, issn])

    log.info('Successfully classified poems from CSV')


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Textblock classifier to poems and other text")
    argparser.add_argument("directory", help="Directory/filename to classify")
    argparser.add_argument("--format", help="Input file format, METS XML or CSV file", default="XML", choices=["XML", "CSV"])
    argparser.add_argument("--newfile", help="Create new CSV file", dest='newfile', action='store_true')
    args = argparser.parse_args()

    log.info('Loading classifier from pickle file')
    clf = joblib.load('svm.pkl')

    log.info('Classifier loaded')

    if args.format == 'CSV':
        classify_csv(args.directory, args.newfile)
    else:
        if args.directory[-1] != '/':
            args.directory += '/'

        classify_xmls(args.directory, args.newfile)
