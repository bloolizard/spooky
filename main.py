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


from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
vectorizer.fit_transform(train_data_df.text)



# Train Data
# id, text, author
train_data_df = pd.read_csv(train_path)

# Test Data
# id, text
test_data_df = pd.read_csv(test_path)

# Sample Submission
# id, EAP, HPL, MWS

# encode x and y
vectorizer = CountVectorizer()
train_data_features = vectorizer.fit_transform(train_data_df.text)
train_data_features = train_data_features.toarray()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
enc_y = le.fit_transform(train_data_df.author)

# from sklearn.preprocessing import OneHotEncoder
# enc = OneHotEncoder()
# enc.fit(enc_y)

from sklearn.preprocessing import LabelBinarizer

encoder = LabelBinarizer()
transfomed_label = encoder.fit_transform(train_data_df.author)
print(transfomed_label)

labels = train_data_df.author

vocab = vectorizer.get_feature_names()

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
n_categories = 3
rnn = RNN(n_input, n_hidden, n_categories)

learning_rate = 0.1
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


criterion = nn.CrossEntropyLoss()

num_epochs = 2

for i in range(len(train_data_features)):
    x = train_data_features[i].astype(float)
    torch_x = torch.from_numpy(x).float()
    sentence = Variable(torch_x)

    label = Variable(torch.from_numpy(enc_y.reshape(-1, 1)[i]))

    optimizer.zero_grad()

    outputs = model(sentence).view(1,3)

    loss = criterion(outputs, label)

    loss.backward()

    optimizer.step()

    iter += 1
    print('Iteration: {}, Loss: {}.'.format(iter, loss.data[0]))




class FeedforwardNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(FeedforwardNeuralNetwork, self).__init__()
        # Linear Function
        self.fc1 = nn.Linear(input_size, hidden_size)

        # Non-linearity
        self.sigmoid = nn.Sigmoid()

        # Linear function (readout)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Linear function
        out = self.fc1(x)
        # Non-linearity
        out = self.sigmoid(out)
        # Linear function (readout)
        out = self.fc2(out)

        return out

model = FeedforwardNeuralNetwork(25068, 1000, 3)


print(len(list(model.parameters())))
print(list(model.parameters())[0].size())

# Convolution 1 Bias: 16 Kernels
print(list(model.parameters())[1].size())

# Convolution 2: 32 Kernels with depth = 16
print(list(model.parameters())[2].size())

# Convolution 2 Bias: 32 Kernels with depth = 16
print(list(model.parameters())[3].size())

# Fully Connected Layer 1
print(list(model.parameters())[4].size())

# Fully Connected Layer Bias
print(list(model.parameters())[5].size())
