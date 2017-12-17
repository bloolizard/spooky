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

path = "~/Developer/spooky/data/"
train_path = path + "train.csv"
test_path = path + "test.csv"


# Train Data
# id, text, author
train_data_df = pd.read_csv(train_path)

# Test Data
# id, text
test_data_df = pd.read_csv(test_path)

# Create Dataset Class
class SpookyTrainDataset(Dataset):
    def __init__(self, csv_file):
        self.spooky_frame = pd.read_csv(csv_file)

        vectorizer = CountVectorizer()
        vectorizer.fit(self.spooky_frame.text)
        vectorize_text = vectorizer.fit_transform(self.spooky_frame.text)
        self.vocab = vectorizer.get_feature_names()
        print(self.vocab)
        self.vectorize_text = vectorize_text.toarray()

        le = LabelEncoder()
        self.enc_author = le.fit_transform(self.spooky_frame.author)

    def __len__(self):
        return len(self.spooky_frame)

    def __getitem__(self, idx):
        sample = {}
        text = self.spooky_frame.text[idx]
        author = self.spooky_frame.author[idx]
        text_tensor = torch.from_numpy(self.vectorize_text[idx].astype("float64")).float()
        sample['text'] = text
        sample['author'] = author
        sample['text_vector'] = self.vectorize_text[idx]
        sample['text_tensor'] = text_tensor
        sample['enc_author'] = self.enc_author[idx]
        return sample

spooky_train_set = SpookyTrainDataset(train_path)
spooky_train_set[0]['text_tensor'].shape

for i in range(len(spooky_train_set)):
    print(spooky_train_set[i])


class FeedForwardNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(FeedForwardNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.sigmoid = nn.Sigmoid()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.sigmoid(out)
        out = self.fc2(out)
        return out

n_input = len(spooky_train_set.vocab)
hidden_dim = 1000
output_dim = 3
model = FeedForwardNN(n_input, hidden_dim, output_dim)

learning_rate = 0.1
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

for i in range(len(spooky_train_set)):
    print(spooky_train_set[i])