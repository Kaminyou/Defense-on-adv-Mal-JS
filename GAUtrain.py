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
from model import CountsNet
from dataset import JSDataset
from attack import Attacker
from defense import train_defense
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    #gaussian noise training
    with open('./good_counts_data.pkl', 'rb') as f:
        good_counts, good_data_idx, good_data_length = pickle.load(f)
    with open('./mal_counts_data.pkl', 'rb') as f:    
        mal_counts, mal_data_idx, mal_data_length = pickle.load(f)
    scaledata = ScaleData(read_from = "./Model/scale_paras.pkl")
    good_counts_scale = scaledata.fit(good_counts, mode = "standardization")
    mal_counts_scale = scaledata.fit(mal_counts, mode = "standardization")

    mu = 0
    sigma_list = [0.0001,0.0005,0.001,0.005,0.01,0.05,0.1,0.5]

    for sigma_now in sigma_list:
        print("==================================")
        print("========GAU NOISE TRAINING========")
        print(sigma_now)
        print("==================================")
        mal_counts_scale_add_noise = mal_counts_scale + np.random.normal(0, sigma_now, (mal_counts_scale.shape[0],mal_counts_scale.shape[1])) 
        all_JS_data = np.vstack((good_counts_scale, mal_counts_scale_add_noise))
        all_JS_label = np.hstack((np.zeros(len(good_counts_scale)),np.ones(len(mal_counts_scale))))
        all_JS_length = np.hstack((good_data_length, mal_data_length))
        all_JS_index = np.hstack((good_data_idx, mal_data_idx))

        X_train, X_test, y_train, y_test, L_train, L_test, ID_train, ID_test = train_test_split(all_JS_data, all_JS_label, all_JS_length, all_JS_index, test_size=0.25, random_state=42)
        train_data_list = [X_train, y_train, L_train, ID_train]
        test_data_list = [X_test, y_test, L_test, ID_test]
        save_name = "DEFENSEMODEL_gau_noise_" + str(sigma_now) +".pt"
        train_defense(train_data_list, test_data_list, batch_size =512, n_epoch = 50, lr=0.01, device="cpu", save_name = save_name)