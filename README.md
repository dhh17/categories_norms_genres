# Categories, norms and genres - critical readings in numbers

This repository contains source codes for finding poems from Finnish newspaper archive, and plotting scripts and keyword charts of the classified poems.
 
The classifier uses scikit-learn, and it is used to classify newspaper text blocks into poems and non-poems. 
Created originally in Digital Humanities Hackathon 2017.

In order to run ´poem_reader.py´, make sure that the requirements listed in ´requirements.txt´ are matched. You need to also have the following subfolders that are used for collected training data:

    - poems
    - nonpoems
    - poemblocks
    - nonpoemblocks