#!/usr/bin/env python3
#  -*- coding: UTF-8 -*-
"""
Reader for the newspaper XML files
"""
import argparse
import glob

from collections import defaultdict

import pandas
from lxml import etree

argparser = argparse.ArgumentParser(description="Newspaper XML parser", fromfile_prefix_chars='@')
argparser.add_argument("dataroot", help="Path to DHH 17 newspapers directory")
args = argparser.parse_args()

data_root = args.dataroot


def read_xml_directory(path):
    """
    Read XML files from path, parse them, and return them as list
    """
    files = glob.glob(path + "*.xml")

    if not files:
        print('No files found for %s' % path)

    xmls = []
    for xmlfile in files:
        with open(xmlfile, 'r') as f:
            parsed = etree.parse(f)
            xmls.append(parsed)

    return xmls


def get_block_texts(xmls, poem_block_ids):
    """
    Find an element by block_id from a list of lxml trees
    """

    def parse_text_lines(lines):
        text = ''
        for line in lines:
            for string in line:
                if 'String' in str(string.tag):
                    text += string.get('CONTENT')
                    #print(text)
                elif 'SP' in str(string.tag):
                    text += ' '
            text += '\n'

        return text

    ns = {'kk': 'kk-ocr'}
    block_xpath = etree.XPath("//kk:TextBlock", namespaces=ns)
    poems = []
    nonpoems = []

    for xml in xmls:
        text_blocks = block_xpath(xml)

        for block in text_blocks:
            if block.get('ID') in poem_block_ids:
                poems.append(parse_text_lines(list(block)))
            else:
                nonpoems.append(parse_text_lines(list(block)))

    return poems, nonpoems


def format_path(doc, issues):
    issue_no = issues.loc[issues['url'] == doc['URL']]['no'].iloc[0]
    date = doc['Date'].to_pydatetime().date()
    formatted = 'newspapers/fin/{y}/{issn}/{issn}_{isodate}_{issue}/alto/'.\
        format(issn=doc['ISSN'], y=date.year, isodate=date.isoformat(), issue=issue_no)

    return formatted


docs = pandas.read_csv('data/docs.csv', sep='\t', parse_dates=[1], dayfirst=True)
issues = pandas.read_csv('data/issue_numbers.csv', sep=',')

doc_ids = defaultdict(list)
for doc in docs.iterrows():
    doc_ids[doc[1]['URL']].append(doc[1]['TextblockID'])

poems = []
others = []

print('Reading XML files...')

for doc in docs.iterrows():
    path = data_root + format_path(doc[1], issues)
    xmls = read_xml_directory(path)
    if not xmls:
        continue

    poem, nonpoem = get_block_texts(xmls, doc_ids[doc[1]['URL']])
    poems += [poem]
    others += [nonpoem]

print('Writing text files')

with open('poems.txt', 'w', newline='') as fp:
    for poem in poems:
        for line in poem:
            fp.write("%s\n" % line)
        fp.write("----------\n\n")

with open('nonpoems.txt', 'w', newline='') as fp:
    for text in others:
        for line in text:
            fp.write("%s\n" % line)
        fp.write("----------\n\n")

