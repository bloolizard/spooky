from __future__ import print_function, division

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

class SpookyTrainDataset(Dataset):
    def __init__(self, csv_file):
        self.spooky_frame = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.spooky_frame)

    def __getitem__(self, idx):
        sample = {}
        text = self.spooky_frame.text[idx]
        author = self.spooky_frame.author[idx]
        sample['text'] = text
        sample['author'] = author
        return sample

    def dataframe(self):
        return self.spooky_frame

spooky_train_set = SpookyTrainDataset(train_path)

