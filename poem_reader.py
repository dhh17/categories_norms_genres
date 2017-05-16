#!/usr/bin/env python3
#  -*- coding: UTF-8 -*-
"""
Reader for the newspaper XML files
"""
import argparse
import glob

import datetime
import pandas
from iso8601 import iso8601
from lxml import etree, objectify

argparser = argparse.ArgumentParser(description="Newspaper XML parser", fromfile_prefix_chars='@')
argparser.add_argument("dataroot", help="Path to DHH 17 newspapers directory")
args = argparser.parse_args()

data_root = args.dataroot


def read_xml_directory(path):
    """
    Read XML files from path, parse them, and return them as list
    """
    files = glob.glob(path + "*.xml")

    xmls = []
    for xmlfile in files:
        with open(xmlfile, 'r') as f:
            parsed = etree.parse(f)
            xmls.append(parsed)

    return xmls


def find_by_block_id(xmls, block_id):
    """
    Find an element by block_id from a list of lxml trees
    """
    ns = {'kk': 'kk-ocr'}
    block_xpath = etree.XPath("//kk:TextBlock[@ID='{id}']".format(id=block_id), namespaces=ns)
    text = ''
    for xml in xmls:
        blocks = block_xpath(xml)
        if blocks:
            lines = list(blocks[0])
            for line in lines:
                for string in line:
                    if 'String' in str(string.tag):
                        text += string.get('CONTENT')
                    elif 'SP' in str(string.tag):
                        text += ' '
                text += '\n'

            return text


def format_path(doc, issues):
    issue_no = issues.loc[issues['url'] == doc['URL']]['no'].iloc[0]
    date = doc['Date'].to_pydatetime().date()
    formatted = 'newspapers/fin/{y}/{issn}/{issn}_{isodate}_{issue}/alto/'.\
        format(issn=doc['ISSN'], y=date.year, isodate=date.isoformat(), issue=issue_no)

    return formatted


docs = pandas.read_csv('docs.csv', sep='\t', parse_dates=[1], dayfirst=True)
issues = pandas.read_csv('issue_numbers.csv', sep=',')

for doc in docs.iterrows():
    path = data_root + format_path(doc[1], issues)
    xmls = read_xml_directory(path)
    text_block = find_by_block_id(xmls, doc[1]['TextblockID'])
    print(text_block)


