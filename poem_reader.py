#!/usr/bin/env python3
#  -*- coding: UTF-8 -*-
"""
Reader for the newspaper XML files
"""
import argparse
import glob

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

    xmls = []
    for xmlfile in files:
        with open(xmlfile, 'r') as f:
            xmls.append(etree.parse(f))

    return xmls


def find_by_block_id(xmls, block_id):
    """
    Find an element by block_id from a list of lxml trees
    """
    block_xpath = etree.XPath("//*[@ID='{id}']".format(id=block_id))
    for xml in xmls:
        elements = block_xpath(xml)
        if elements:
            return elements[0]


some_dir = data_root + 'newspapers/fin/1854/1457-4616/1457-4616_1854-08-01_31/alto/'
xmls = read_xml_directory(some_dir)

print(etree.tostring(find_by_block_id(xmls, 'P2_TB00001')))


