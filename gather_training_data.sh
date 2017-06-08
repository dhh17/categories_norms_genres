#!/usr/bin/env bash

rm poemblocks/*
rm nonpoemblocks/*
python poem_reader.py ~/dhh17data/newspapers/
# python split_to_files.py data/additional_poemtexts.txt
# ls -l data/additional_poemtexts_* | wc -l
# mv data/additional_poemtexts_* poemblocks/
