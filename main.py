from __future__ import division
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.datasets import fetch_20newsgroups
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import f1_score
from numpy.random import randint
import logging
import sys
import operator
import math
from sklearn.multiclass import OneVsRestClassifier
from collections import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from nltk.corpus import wordnet as wn
import re
from itertools import chain
from sklearn.utils import shuffle
import nltk
import string
import os
from nltk.stem.porter import PorterStemmer
from string import maketrans
from sklearn.cluster import KMeans
import inspect
from multiprocessing import Process
from nltk import word_tokenize
from nltk.corpus import stopwords
from datetime import datetime
import pickle
from sklearn.svm import SVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.calibration import calibration_curve
import itertools as it
from numpy import linalg as LA

train = fetch_20newsgroups(subset='train')
test = fetch_20newsgroups(subset='test')

train_data = train.data
train_targets = train.target

test_data = test.data
test_targets = test.target

baseline_noactive_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', SVC(kernel='linear', decision_function_shape='ovo', verbose=True)),
])

print'start fit'
baseline_noactive_clf.fit(train_data, train_targets)
print 'fitted'
print 'start predict'
predicted = baseline_noactive_clf.predict(test_data)
print 'predicted'
score = f1_score(test_targets, predicted, average='macro')
print 'res:', score