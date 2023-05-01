import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import torch
from torch.nn.utils.rnn import pad_sequence


def load_data(x_path):
    return pd.read_csv('text_x.csv')


def split_data(x, y, split=0.8):
    return train_test_split(x, y, split)


def preprocess_x(df):
    # Your code here
    return data
