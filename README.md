# wiki-classifier

A simple package for scraping Wikipedia pages from particular categories and building a classifier.

## Usage

#### scrape_and_build.py

This file will scrape wikipedia pages from categories specified in a text file and build a classifier from them.
If you haven't downloaded the data already, you can run 

`python scrape_and_build.py --categories=categories.txt`

If you've already run scraping, you can simply run

`python scrape_and_build.py --use_cached_data`

to rebuild the classifier.

#### classify.py

This file takes a url and prints out the probabilities of belonging to categories specified in the classifier building above. You can run this with

`python classify.py --url=http://www.blah.com`

You can also specify which classifier file to use with the `--classifier` option.
