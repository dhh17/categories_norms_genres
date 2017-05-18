#!/usr/bin/env python3
#  -*- coding: UTF-8 -*-
import argparse

argparser = argparse.ArgumentParser(description="Split text file rows to separate files")

argparser.add_argument("filename", help="Filename to split")
args = argparser.parse_args()

with open(args.filename, 'r') as input:
    for index, line in enumerate(input):
        with open('{prefix}_{id}.txt'.format(id=index, prefix=args.filename.split('.')[0]), 'w') as output:
            output.write(line)
