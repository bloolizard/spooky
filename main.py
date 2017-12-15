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
from torch.autograd import Variable

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

sentences = train_data_df.text
sentence = sentences[0]
for word in sentence:
    words = sentence.split(" ")
    print(word)

for sentence in sentences:
    print sentence

def sentenceToVec(sentence):
    vec = 0
    return vec

all_categories = []
n_categories = len(all_categories)

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return Variable(torch.zeros(1, self.hidden_size))

n_hidden = 128
n_input = len(vocab)
rnn = RNN(n_input, n_hidden, n_categories)

learning_rate = 0.1
optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)

criterion = nn.NLLLoss()

num_epochs = 2

for epoch in range(num_epochs):
    for sentence in len(sentences):
        sentences = Variable(sentence)
        labels = Variable(labels)

        optimizer.zero_grad()

        outputs = rnn(sentences)

        loss = criterion(outputs)

        loss.backward()

        optimizer.step()

        iter += 1
        if iter % 100 == 0:
            print('Iteration: {}, Loss: {}.'.format(iter, loss.data[0]))

