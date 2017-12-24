from __future__ import print_function, division

import numpy
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

from sklearn.preprocessing import LabelBinarizer
encoder = LabelBinarizer()
transfomed_label = encoder.fit_transform(train_data_df.author)
print(transfomed_label)

class SpookyTrainDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.spooky_frame = pd.read_csv(csv_file)

        vectorizer = CountVectorizer()
        vectorizer.fit(self.spooky_frame.text)
        vectorize_text = vectorizer.fit_transform(self.spooky_frame.text)
        self.vocab = vectorizer.get_feature_names()
        self.vectorize_text = vectorize_text.toarray()

        le = LabelEncoder()
        self.enc_author = le.fit_transform(self.spooky_frame.author)

        binary_encoder = LabelBinarizer()
        self.binary_label = binary_encoder.fit_transform(self.spooky_frame.author)
        self.trasnform = transform

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
        sample['enc_author'] = torch.from_numpy(self.enc_author.reshape(-1,1)[idx])
        sample['binary_label'] = torch.from_numpy(self.binary_label[idx])
        return sample

spooky_train_set = SpookyTrainDataset(train_path)

class FeedForwardNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(FeedForwardNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.sigmoid = nn.Sigmoid()
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.Softmax()

    def forward(self, x):
        out = self.fc1(x)
        out = self.sigmoid(out)
        out = self.fc2(out)
        out = self.softmax(out)
        return out

n_input = len(spooky_train_set.vocab)
hidden_dim = 1000
output_dim = 3
model = FeedForwardNN(n_input, hidden_dim, output_dim)

learning_rate = 0.1
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

iter = 0
for i in range(10):
    sample = spooky_train_set[i]
    sentence = Variable(sample['text_tensor'])
    label = Variable(sample['enc_author'])

    optimizer.zero_grad()

    outputs = model(sentence).view(1,3)
    print(outputs.size())
    print(label.size())
    loss = criterion(outputs, label)

    loss.backward()
    optimizer.step()
    iter += 1
    print('Iteration: {}, Loss: {}.'.format(iter, loss.data[0]))

# todo: need to turn this into probability
outputs.data
_, predicted = torch.max(outputs.data, 1)
top3_prob, top3_label = torch.topk(outputs.data, 3)


iter = 0
for i in range(10):
    sample = spooky_train_set[i]
    sentence = Variable(sample['text_tensor'])
    label = Variable(sample['enc_author'])

    optimizer.zero_grad()

    outputs = model(sentence).view(1,3)
    print(outputs.size())
    print(label.size())
    loss = criterion(outputs, label)

    loss.backward()
    optimizer.step()
    iter += 1
    print('Iteration: {}, Loss: {}.'.format(iter, loss.data[0]))

# Sample Submission
# id, EAP, HPL, MWS
def create_output_csv():
    columns = ["id", "EAP", "HPL", "MWS"]
    df = pd.DataFrame(columns=columns)
    df.loc[1] = [1,95,2,3]
    df.to_csv("output.csv", index=False)

create_output_csv()

probs = []
for i in range(10):
    sample = spooky_train_set[i]
    sentence = Variable(sample['text_tensor'])
    label = Variable(sample['enc_author'])

    outputs = model(sentence).view(1,3)
    probs.append(outputs.data[0].numpy())

# Sample Submission
# id, EAP, HPL, MWS
def create_output_csv():
    columns = ["id", "EAP", "HPL", "MWS"]
    df = pd.DataFrame(columns=columns)
    df.loc[1] = [1,95,2,3]
    df.to_csv("output.csv", index=False)

create_output_csv()

columns = ["id", "EAP", "HPL", "MWS"]
df = pd.DataFrame(columns=columns)
df.append(probs)
df.to_csv("output.csv", index=False)

df.insert(0, 'id', train_data_df.id)

# drop all values from pandas
df = df.iloc[0:0]

probs = []
for i in range(10):
    sample = spooky_train_set[i]
    sentence = Variable(sample['text_tensor'])
    label = Variable(sample['enc_author'])

    outputs = model(sentence).view(1,3)
    probs.append(outputs.data[0].numpy())

for i in range(len(probs)):
    df.loc[i] = [train_data_df.id[i], probs[i][0], probs[i][1], probs[i][2]]

df