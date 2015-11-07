#!/usr/bin/env python

import numpy as np
from bs4 import BeautifulSoup
import urllib
import argparse
import sys
import pickle
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.base import BaseEstimator

'''
A script for scraping wiki articles and building a classifier.

One drawback is that the output file names are currently hard-coded. This would be a potential improvement.
Also allowing the user to specify whether they want multi-label classification or not.
'''

class DenseTransform(BaseEstimator):
    '''
    A simple class for transforming a sparse scipy matrix to dense for sklearn pipeline usage
    '''

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.toarray()

    def fit_transform(self, X, y=None):
        return X.toarray()


def parse_ind_page(page):
    '''
    Helper function for extracting the text fron an individual wikipedia article's HTML.
    Returns only the text from the table of contents and any paragraphs
    '''
    article = BeautifulSoup(page)
    toc = article.find_all("div", {"id" : "toc"})

    #If there is a table of contents, includei t
    text = ""
    if len(toc) > 0:
        text = toc[0].text

    #Include the text from all the paragraphs
    paragraphs = article.find_all("p") 
    for p in paragraphs:
        text += p.text

    return text

def parse_category(page, cat_num, target_dict, text_dict):
    '''
    Helper function to parse all the pages for a particular category.
    Takes in an open URL object, a category number (for target labeling), a dictionary of targets to 
    add to, and a dictionary of text to add to. The keys for the dictionary are the article names.
    '''
    bs_obj = BeautifulSoup(page)
    article_list = bs_obj.find_all("div", {"id" : "mw-pages"})[0]

    links = article_list.find_all("a")
    prefix = "https://en.wikipedia.org"
    ct = 0
    for link in links:
        href = link["href"]
        title = link["title"]
        text = link.text

        # Some category pages span multiple categories, so we call the function recursively
        # if this is the case
        if text == "next page":
            text_dict, target_dict = parse_category(urllib.urlopen(prefix+href), cat_num, target_dict, text_dict)
        else:
            # If there is already an entry in the dictionary, we add the new category label.
            # Otherwise, we put a new entry in the dictionary.
            # This allows for multi-label classification later
            if title in text_dict:
                curr_cats = target_dict[title]
                if not cat_num in curr_cats:
                    curr_cats += [cat_num]
                    target_dict[title] = curr_cats
            else:
                if title.startswith("Wikipedia:") or title.startswith("Category:"):
                    continue
                page_text = parse_ind_page(urllib.urlopen(prefix+href))
                text_dict[title] = page_text
                target_dict[title] = [cat_num]

                ct += 1

    return text_dict, target_dict

def cat_list_to_vector(cat_list, n_topics):
    '''
    Helper function which takes a list of categories and turns it into a multi-label vector
    '''
    ret = np.zeros((1, n_topics))
    for c in cat_list:
        ret[0, c] = 1

    return ret

def main():
    ops = options()

    target_dict = {}
    text_dict = {}
    ind_cat_map = {}
    cat_ind_map = {}

    # If we haven't already downloaded the wiki data, do that first.
    if not ops.use_cached_data:
        if ops.categories is None:
            sys.exit("Error in %s: %s" % (__file__, "Please provide a filename for the list of categories with argument --categories or the option --use_cached_data"))

        print "Data not cached...downloading!"
        cat_file = None

        # Read in categories from a text file and call the parsing helper
        try:
            cat_file = open(ops.categories, "r")
        except:
            sys.exit("Error in %s: %s" % (__file__, "Categories file could not be opened."))

        cat_num = 0
        for line in cat_file:
            url_cat = line.replace(" ", "_")
            curr_url = "https://en.wikipedia.org/wiki/Category:" + url_cat

            open_page = None
            try:
                open_page = urllib.urlopen(curr_url)
            except:
                sys.exit("Error in %s: %s" % (__file__, "Couldn't find page for category " + line))
                continue


            print "Parsing category " + line
            text_dict, target_dict = parse_category(open_page, cat_num, target_dict, text_dict)
            ind_cat_map[cat_num] = line
            cat_ind_map[line] = cat_num
            cat_num += 1

        # Dump the dictionary of text and labels for each article, as well as
        # a dictionary for converting between numerical target label and category name
        # (and vice versa)
        pickle.dump(text_dict, open("text_dict.pkl", "w"))
        pickle.dump(target_dict, open("target_dict.pkl", "w"))
        pickle.dump(ind_cat_map, open("ind_cat_map.pkl", "w"))
        pickle.dump(cat_ind_map, open("cat_ind_map.pkl", "w"))
    else: #We already downloaded the wiki data, so let's load it.
        print "Using cached articles..."
        try:
            text_dict = pickle.load(open("text_dict.pkl", "r"))
            target_dict = pickle.load(open("target_dict.pkl", "r"))
            ind_cat_map = pickle.load(open("ind_cat_map.pkl", "r"))
            cat_ind_map = pickle.load(open("cat_ind_map.pkl", "r"))
        except:
            sys.exit("Error in %s: %s" % (__file__, "Couldn't find cached data files! Please run without --use_cached_data to download the pages"))


    print "Training classifier..."
    # Now we have the data, so let's build the classifier. 
    pipeline = make_pipeline(TfidfVectorizer(min_df = 10, stop_words='english'), DenseTransform(), OneVsRestClassifier(GradientBoostingClassifier()))
    n_topics = len(cat_ind_map.keys())
    documents = []
    targets = np.zeros((len(text_dict.keys()), n_topics))

    ct = 0
    for k in sorted(text_dict.keys()):
        documents.append(text_dict[k])
        targets[ct, :] = cat_list_to_vector(target_dict[k], n_topics)
        ct += 1

    pipeline = pipeline.fit(documents, targets)

    pickle.dump(pipeline, open("classifier.pkl", "w"))

    print "All done!"

    return 0


def options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--categories")
    parser.add_argument("--use_cached_data", action='store_true')
    return parser.parse_args()

if __name__ == "__main__":
    main()