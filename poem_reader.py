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

ns = {'kk': 'kk-ocr'}
block_xpath = etree.XPath("//kk:TextBlock", namespaces=ns)


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


def parse_text_lines(lines):
    """
    Parse text lines of a text block
    """
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


def get_block_texts(xmls, poem_block_ids):
    """
    Find an element by block_id from a list of lxml trees
    """

    poems = []
    nonpoems = []

    for xml in xmls:
        text_blocks = block_xpath(xml)

        for block in text_blocks:
            text = parse_text_lines(list(block))
            text = text.replace('w', 'v').replace('W', 'V')

            if block.get('ID') in poem_block_ids:
                poems.append(text)
            else:
                nonpoems.append(text)

    return poems, nonpoems


def format_path(doc, issues):
    try:
        issue_no = issues.loc[issues['url'] == doc['URL']]['no'].iloc[0]
    except:
        print('No issue number found for %s %s %s' % (doc['Date'], doc['Paper'], doc['TextblockID']))
        issue_no = '*'

    date = doc['Date'].to_pydatetime().date()
    formatted = 'newspapers/fin/{y}/{issn}/{issn}_{isodate}_{issue}/alto/'.\
        format(issn=doc['ISSN'], y=date.year, isodate=date.isoformat(), issue=issue_no)

    return formatted

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Newspaper XML parser", fromfile_prefix_chars='@')
    argparser.add_argument("dataroot", help="Path to DHH 17 newspapers directory")
    args = argparser.parse_args()

    data_root = args.dataroot

    docs = pandas.read_csv('data/docs.csv', sep='\t', parse_dates=[1], dayfirst=True)
    issues = pandas.read_csv('data/issue_numbers.csv', sep=',')

    doc_ids = defaultdict(list)
    for doc in docs.iterrows():
        doc_ids[doc[1]['URL']].append(doc[1]['TextblockID'])

    print('Reading XML files...')
    path = None

    for doc in docs.iterrows():
        previous_path = path
        path = data_root + format_path(doc[1], issues)
        if path == previous_path:
            continue
        xmls = read_xml_directory(path)
        if not xmls:
            continue

        poem, nonpoem = get_block_texts(xmls, doc_ids[doc[1]['URL']])

        date = doc[1]['Date'].to_pydatetime().date()
        with open('poems/{journal}_{date}.txt'.format(journal=doc[1]['Paper'], date=date), 'w', newline='') as fp:
            for line in poem:
                fp.write("%s\n" % line)

        with open('nonpoems/{journal}_{date}.txt'.format(journal=doc[1]['Paper'], date=date), 'w', newline='') as fp:
            for line in nonpoem:
                fp.write("%s\n" % line)

        date = doc[1]['Date'].to_pydatetime().date()
        for (i, block) in enumerate(poem):
            with open('poemblocks/{journal}_{date}_{i}.txt'.format(journal=doc[1]['Paper'], date=date, i=i), 'w', newline='') as fp:
                fp.write(block)

        for i, block in enumerate(nonpoem):
            with open('nonpoemblocks/{journal}_{date}_{i}.txt'.format(journal=doc[1]['Paper'], date=date, i=i), 'w', newline='') as fp:
                fp.write(block)
