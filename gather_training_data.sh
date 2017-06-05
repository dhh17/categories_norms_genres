#!/usr/bin/env bash

python poem_reader.py ~/dhh17data/newspapers/
python split_to_files.py data/additional_poemblocks.txt
ls -l data/additional_poemblocks_* | wc -l
mv data/additional_poemblocks_* poemblocks/
