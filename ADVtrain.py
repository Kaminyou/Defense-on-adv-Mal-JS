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
    #adversarial training
    # prepare normal data
    
    with open('./good_counts_data.pkl', 'rb') as f:
        good_counts, good_data_idx, good_data_length = pickle.load(f)
    with open('./mal_counts_data.pkl', 'rb') as f:    
        mal_counts, mal_data_idx, mal_data_length = pickle.load(f)
    scaledata = ScaleData(read_from = "./Model/scale_paras.pkl")
    good_counts_scale = scaledata.fit(good_counts, mode = "standardization")
    mal_counts_scale = scaledata.fit(mal_counts, mode = "standardization")
    all_JS_data = np.vstack((good_counts_scale, mal_counts_scale))
    all_JS_label = np.hstack((np.zeros(len(good_counts_scale)),np.ones(len(mal_counts_scale))))
    all_JS_length = np.hstack((good_data_length, mal_data_length))
    all_JS_index = np.hstack((good_data_idx, mal_data_idx))

    X_train, X_test, y_train, y_test, L_train, L_test, ID_train, ID_test = train_test_split(all_JS_data, all_JS_label, all_JS_length, all_JS_index, test_size=0.25, random_state=42)
    test_data_list = [X_test, y_test, L_test, ID_test]
    Adv_path = "./Adv_example/"

    for adv_examples_file in os.listdir(Adv_path):
        adv_examples_data_path = os.path.join(Adv_path, adv_examples_file)
        print("============================")
        print("========ADV TRAINING========")
        print(adv_examples_file)
        print("============================")
        with open(adv_examples_data_path, 'rb') as f:
            adv_examples_data = pickle.load(f)
        adv_status_all = []
        adv_ori_data_all = []
        adv_adv_data_all = []
        adv_label_all = []
        adv_length_all = []
        adv_ID_all = []
        adv_mal_all = []
        adv_examples_data = adv_examples_data[:8000] #########################################
        for data in adv_examples_data:
            if data["status"] == "success":
                adv_status_all.append(data["status"])
                adv_ori_data_all.append(data["data"])
                adv_adv_data_all.append(data["adv_data"])
                adv_label_all.append(int(data["label"][0]))
                adv_length_all.append(int(data["length"][0]))
                adv_ID_all.append(data["ID"][0])
                adv_mal_all.append(data["adv_data"])
            else:
                adv_status_all.append(data["status"])
                adv_ori_data_all.append(data["data"])
                adv_adv_data_all.append(data["adv_data"])
                adv_label_all.append(int(data["label"][0]))
                adv_length_all.append(int(data["length"][0]))
                adv_ID_all.append(data["ID"][0])
                adv_mal_all.append(data["data"])

        #concat
        X_train_concat = np.vstack((good_counts_scale, adv_mal_all))
        y_train_concat = np.hstack((np.zeros(len(good_counts_scale)), np.ones(len(adv_adv_data_all))))
        L_train_concat = np.hstack((good_data_length, adv_length_all))
        ID_train_concat = np.hstack((good_data_idx, adv_ID_all))

        train_data_list = [X_train_concat, y_train_concat, L_train_concat, ID_train_concat]

        save_name = "DEFENSEMODEL_adv_training_" + adv_examples_file +".pt"
        train_defense(train_data_list, test_data_list, batch_size =512, n_epoch = 50, lr=0.01, device="cpu", save_name = save_name)