# import code; code.interact(local=dict(globals(), **locals()))  # debugger

import torch
from data import *
from model import *
import random
import time
import math

n_hidden = 128
n_epochs = 1000
print_every = 1
plot_every = 1000
learning_rate = 0.005 # If you set this too high, it might explode. If too low, it might not learn

categories = ['EAP', 'HPL', 'MWS']

def multiclass_logloss(actual, predicted, eps=1e-15):
    """Multi class version of Logarithmic Loss metric.
    :param actual: Array containing the actual target classes
    :param predicted: Matrix with class predictions, one probability per class
    """
    # Convert 'actual' to a binary array if it's not already:
    if len(actual.shape) == 1:
        actual2 = np.zeros((actual.shape[0], predicted.shape[1]))
        for i, val in enumerate(actual):
            actual2[i, val] = 1
        actual = actual2

    clip = np.clip(predicted, eps, 1 - eps)
    rows = actual.shape[0]
    vsota = np.sum(actual * np.log(clip))
    return -1.0 / rows * vsota

def categoryFromOutput(output):
    top_n, top_i = output.data.topk(1) # Tensor out of Variable with .data
    category_i = top_i[0][0]
    return categories[category_i], category_i

def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

def randomTrainingPair():
    category = randomChoice(categories)
    line = randomChoice(category_lines[category])
    category_tensor = Variable(torch.LongTensor([categories.index(category)]))
    line_tensor = Variable(lineToTensor(line))
    return category, line, category_tensor, line_tensor

rnn = RNN(100, n_hidden, len(categories))
optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)
criterion = nn.NLLLoss()

def train(category_tensor, line_tensor):
    hidden = rnn.initHidden()
    rnn.zero_grad()
    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    loss = criterion(output, category_tensor[i])
    loss.backward()

    optimizer.step()

    return output, loss.data[0]

# Keep track of losses for plotting
current_loss = 0
all_losses = []

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

start = time.time()

xtrain_glove, xvalid_glove, ytrain, yvalid = preprocess()

for epoch in range(1, n_epochs + 1):
    output, loss = train(ytrain, xtrain_glove)
    current_loss += loss

    # Print epoch number, loss, name and guess
    if epoch % print_every == 0:
        # guess, guess_i = categoryFromOutput(output)
        # correct = '✓' if guess == category else '✗ (%s)' % category
        print('%d %d%% (%s) %.4f' % (epoch, epoch / n_epochs * 100, timeSince(start), loss))

    # Add current loss avg to list of losses
    if epoch % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0

torch.save(rnn, 'model.pth')
