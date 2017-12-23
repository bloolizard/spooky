import torch
import torchtext.vocab as vocab
import glob
import unicodedata
import string
import pandas as pd
import numpy as np
import json
from torch.autograd import Variable
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from nltk import word_tokenize
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
sample = pd.read_csv('data/sample_submission.csv')

train.head()
test.head()
sample.head()

labels = ['EAP', 'HPL', 'MWS']

lbl_enc = preprocessing.LabelEncoder()
y = lbl_enc.fit_transform(train.author.values)

xtrain, xvalid, ytrain, yvalid = train_test_split(train.text.values, y,
                                                  stratify=y,
                                                  random_state=42,
                                                  test_size=0.1, shuffle=True)

print (xtrain.shape)
print (xvalid.shape)

glove = vocab.GloVe(name='6B', dim=100)

print('Loaded {} words'.format(len(glove.itos)))

def save_obj(obj, name):
    with open(name + '.json', 'w') as f:
        json.dump(obj, f, indent=4)

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return json.load(f)

def get_word(word):
    return glove.vectors[glove.stoi[word]]

embeddings_index = {}


def sent2vec(s):
    words = str(s).lower()
    words = word_tokenize(words)
    words = [w for w in words if not w in stop_words]
    words = [w for w in words if w.isalpha()]
    M = []
    for w in words:
        try:
            M.append(get_word(w))
        except:
            continue
    M = np.array(M)
    v = M.sum(axis=0)
    if type(v) == torch.FloatTensor:
        return v.numpy().reshape(1, 100) / np.sqrt((v ** 2).sum())
    return np.zeros([1, 100])
    # import code; code.interact(local=dict(globals(), **locals()))  # debugger
    # if type(v) != np.ndarray:
    #     return np.zeros([1, 300])



# create sentence vectors using the above function for training and validation set
def preprocess():
    xtrain_glove = [sent2vec(x) for x in tqdm(xtrain)]
    xvalid_glove = [sent2vec(x) for x in tqdm(xvalid)]
    xtrain_glove = Variable(torch.FloatTensor(np.array(xtrain_glove)))
    xvalid_glove = Variable(torch.FloatTensor(np.array(xvalid_glove)))
    return xtrain_glove, xvalid_glove, Variable(torch.LongTensor(ytrain)), Variable(torch.LongTensor(yvalid))
