import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pickle
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
import seaborn as sns
from util import *

if __name__ == '__main__':
    with open('./good_data.pkl', 'rb') as f:
        good_data, good_data_idx, good_data_length = pickle.load(f)
    with open('./mal_data.pkl', 'rb') as f:    
        mal_data, mal_data_idx, mal_data_length = pickle.load(f)

    good_counts = preprocessing_bags_of_word(good_data)
    mal_counts = preprocessing_bags_of_word(mal_data)
    with open('./good_counts_data.pkl', 'wb') as f:
        pickle.dump([good_counts, good_data_idx, good_data_length], f)
    with open('./mal_counts_data.pkl', 'wb') as f:    
        pickle.dump([mal_counts, mal_data_idx, mal_data_length], f)