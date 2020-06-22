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
    good_data, good_data_idx, good_data_length = read_data("./good_script/")
    mal_data, mal_data_idx, mal_data_length = read_data("./mal_script/")
    with open('./good_data.pkl', 'wb') as f:
        pickle.dump([good_data, good_data_idx, good_data_length], f)
    with open('./mal_data.pkl', 'wb') as f:
        pickle.dump([mal_data, mal_data_idx, mal_data_length], f)

    length_distribution(good_data, saving_name = "./Naive_JS_distribution.png")
    length_distribution(mal_data, saving_name = "./Mal_JS_distribution.png")