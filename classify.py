#!/usr/bin/env python

import numpy as np
from bs4 import BeautifulSoup
import urllib
import argparse
import sys
import pickle
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.multiclass import OneVsRestClassifier
from scrape_and_build import parse_ind_page, DenseTransform

'''
A script for classifying a specified wiki URL with a pre-built classifier
'''

def main():
    ops = options()

    #Open the given URL
    url = None
    if ops.url is None:
        sys.exit("Error in %s: %s" % (__file__, "Please provide a wiki url with argument --url"))
    else:
        try:
            url = urllib.urlopen(ops.url)
        except:
            sys.exit("Error in %s: %s" % (__file__, "URL could not be opened!"))


    #Get the classifier
    classifier_name = "classifier.pkl"
    if ops.classifier is None:
        print "Using default classifier filename " + classifier_name
    else:
        classifier_name = ops.classifier


    classifier = None

    try:
        classifier = pickle.load(open(classifier_name, 'r'))
    except:
        sys.exit("Error in %s: %s" % (__file__, "Couldn't find classifier file " + classifier_name))



    # Parse the wiki's text and get probabilities from the classifier
    document = parse_ind_page(url)
    probs = classifier.predict_proba([document])
    probs = probs/np.sum(probs) #Normalize the probabilities

    try:
        ind_cat_map = pickle.load(open("ind_cat_map.pkl", "r"))

        print "====Probabilities of each category===="
        for i in xrange(probs.shape[1]):
            print "%s : %f %%" % (ind_cat_map[i], probs[0, i]*100)
    except:
        print "Couldn't find index to category map...dumping raw probabilities"
        print probs


    return 0


def options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--classifier")
    parser.add_argument("--url")
    return parser.parse_args()

if __name__ == "__main__":
    main()