# AuthFilt

This repository gather functions and classes allowing to apply a highly scalable and efficient author filtering process on any corpus. The goal of the filtering process is to validate the *filtering assumption* stating that removing the most informative sentences about the identity of authors in a reference corpus (i.e. containing the most author-consistent sequences of words) allows to enhance a representation learning aiming to embed document in a stylometric space. The most informative sentences are those containing author-specific word sequences, i.e. word sequences that are used frequently by one author and very little by the rest of the authors in the corpus. The filtering process is based on the TFIDF weighting which is designed to remove terms which are too peculiar from certain authors from the reference corpus. Targeted terms are those having a high frequency in documents of individual authors and having a low inverse document frequency, i.e. those that are rare in the corpus.

## Requirements

Use this script to install all dependencies : <https://github.com/hayj/Bash/blob/master/hjupdate.sh>

## Apply the filtering process

To process your corpus, you need to follow [this notebook](https://github.com/hayj/AuthFilt/blob/master/authfilt/test/demo.ipynb) that use `buckets.py`. You will find each step and comments when a distributed computing can be done. 

## Other ressources

All datasets and trained models are available at <http://212.129.44.40/DeepStyle/>.

The TFIDF computation with dichotomic search is available in a dependency. This part of the project is not yet pushed to keep the repo anonymous.