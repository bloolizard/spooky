from __future__ import print_function, division
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import nltk
from sklearn.model_selection import train_test_split
import torchtext.vocab as vocab
from nltk import word_tokenize
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

from tqdm import tqdm

path = "~/Developer/spooky/data/"
train_path = path + "train.csv"
test_path = path + "test.csv"

train_data_df = pd.read_csv(train_path)
test_data_df = pd.read_csv(test_path)
le = LabelEncoder()
y = le.fit_transform(train_data_df.author.values)

categories = le.classes_
def lookup_category(index):
    return categories[index]

X_train, X_test, y_train, y_test = train_test_split(train_data_df.text,
                                                    y,
                                                    train_size=0.10,
                                                    test_size=0.10,
                                                    random_state=42,
                                                    shuffle=False)
X_train, y_train

X_train[10], lookup_category(y_train[10])

# bag of words
word_to_ix = {}
for sent in X_train:
    split_sent = sent.split(" ")
    for word in split_sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
print(word_to_ix)
vocab_size = len(word_to_ix)
num_categories = len(categories)

def make_bow_vector(sentence, word_to_ix):
    vec = torch.zeros(len(word_to_ix))
    split_sent = sentence.split(" ")
    for word in split_sent:
        if word in word_to_ix:
            vec[word_to_ix[word]] += 1
    return vec.view(1,-1)

def make_label_vector(label):
    return torch.LongTensor([label])


class BoWClf(nn.Module):
    def __init__(self, num_labels, vocab_size):
        super(BoWClf, self).__init__()
        self.linear = nn.Linear(vocab_size, num_labels)

    def forward(self, bow_vec):
        return F.log_softmax(self.linear(bow_vec))

model = BoWClf(num_categories, vocab_size)
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)
num_epochs = 10000
for epoch in tqdm(range(num_epochs)):
    for i, sent in enumerate(X_train):
        model.zero_grad()

        bow_vec = Variable(make_bow_vector(sent, word_to_ix))
        label = Variable(make_label_vector(y_train[i]))
        outputs = model(bow_vec)
        loss = loss_function(outputs, label)
        loss.backward()
        optimizer.step()

torch.save(model, 'model2.pth')
model = torch.load('model2.pth')

# predict entire csv
# predicted csv
columns = ["id", "EAP", "HPL", "MWS"]
df = pd.DataFrame(columns=columns)
for i, sent in tqdm(enumerate(test_data_df.text)):
    bow_vec = Variable(make_bow_vector(sent, word_to_ix))
    outputs = model(bow_vec)
    _, predicted = torch.max(outputs.data, 1)
    softmax_probs = nn.Softmax()(Variable(outputs.data))
    eap_prob = softmax_probs.data.tolist()[0][0]
    hpl_prob = softmax_probs.data.tolist()[0][1]
    mws_prob = softmax_probs.data.tolist()[0][2]
    df.loc[i] = [test_data_df.id[i], eap_prob, hpl_prob, mws_prob]
df.to_csv("output5.csv", index=False)

