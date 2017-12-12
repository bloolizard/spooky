import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

path = "/home/edwin/Developer/spooky/data/"
train_path = path + "train.csv"
test_path = path + "test.csv"

# Train Data
# id, text, author
train_data_df = pd.read_csv(train_path)

# Test Data
# id, text
test_data_df = pd.read_csv(test_path)

# Sample Submission
# id, EAP, HPL, MWS