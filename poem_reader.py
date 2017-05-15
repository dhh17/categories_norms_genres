#!/usr/bin/env python3
#  -*- coding: UTF-8 -*-
"""
Reader for the newspaper XML files
"""
import argparse

from lxml import etree

argparser = argparse.ArgumentParser(description="Newspaper XML parser", fromfile_prefix_chars='@')

argparser.add_argument("dataroot", help="Path to DHH 17 newspapers directory")

args = argparser.parse_args()
data_root = args.dataroot

with open(data_root + 'newspapers/fin/1854/1457-4616/1457-4616_1854-08-01_31/alto/1457-4616_1854-08-01_31_001.xml', 'r') as f:
    tree = etree.parse(f)

root = tree.getroot()
print(root.tag)
