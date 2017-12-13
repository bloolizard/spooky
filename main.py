import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import nltk

path = "~/Developer/spooky/data/"
train_path = path + "train.csv"
test_path = path + "test.csv"

from __future__ import division

from sklearn.feature_extraction.text import CountVectorizer

# Train Data
# id, text, author
train_data_df = pd.read_csv(train_path)

# Test Data
# id, text
test_data_df = pd.read_csv(test_path)

# Sample Submission
# id, EAP, HPL, MWS

test_data_df

word_to_ix = {}


total_words = train_data_df.text.str.split(" ")

len(total_words)

sentence = train_data_df.text[0].split(" ")
len(sentence)
len(set(sentence))
len(set(sentence)) / len(sentence)

sentence = train_data_df.text[0].split(" ")
vocab = set(sentence)
vocab

word_to_ix = {word: i for i, word in enumerate(vocab) }

# using NLTK
sentence = train_data_df.text[0]
first_sentence = nltk.word_tokenize(sentence)

stopwords = nltk.corpus.stopwords.words('english')
sentence_filtered = [word for word in first_sentence if word.lower() not in stopwords]

vectorizer = CountVectorizer()
sentence_transform = vectorizer.fit_transform(sentence_filtered)
sentence_transform

