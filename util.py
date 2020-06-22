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

def read_data(path):
    data_all = []
    data_idx = []
    data_length = []
    total_count = 0
    accept_count = 0
    error_count = 0
    for i, txt in enumerate(os.listdir(path)):
        print(f"process {i+1}/{len(os.listdir(path))}", end="\r")
        total_count += 1
        data_path = os.path.join(path, txt)
        data = []
        try:
            with open(data_path) as f:
                for line in f:
                    for token in line:
                        if ord(token) >= 256:
                            pass
                        else:
                            data.append(ord(token))
            data_all.append(data)
            data_idx.append(txt)
            data_length.append(len(data))
            accept_count += 1
        except Exception as e:
            print(f"ERROR in {data_path}")
            error_count += 1
    print("Finish                                       ")
    print(f"Total: {total_count}")
    print(f"Accepted: {accept_count}")
    print(f"Error: {error_count}")
    return data_all, data_idx, data_length

def length_distribution(data_all, saving_name = "distribution"):
    data_len = [len(data) for data in data_all]
    plt.figure(figsize=(20, 4))
    plt.hist(data_len, bins = 500, log=True)
    #plt.show()
    plt.savefig(saving_name)

def preprocessing(data_all, padding=250000):
    post_data = []
    for i, data in enumerate(data_all):
        print(f"process {i+1}/{len(data_all)}", end="\r")
        if len(data) > padding:
            data = data[:padding]
            post_data.append(data)
        else:
            data = np.pad(data, (0, (padding-len(data))), 'constant', constant_values=0)
            post_data.append(data)
    post_data = np.array(post_data) 
    print(f"Shape: {post_data.shape}")
    return post_data

def _bag_of_word(single_data):
    counts = np.zeros(256, dtype = int)
    idx, num = np.unique(single_data,return_counts = True)
    counts[idx] = num
    return counts

def preprocessing_bags_of_word(data_all):
    counts_all = []
    for data in data_all:
        counts = _bag_of_word(data)
        counts_all.append(counts)
    counts_all = np.array(counts_all)
    return counts_all

class ScaleData(object):
    def __init__(self, training_data=None, read_from = None):
        
        if read_from != None:
            with open(read_from, 'rb') as f:
                readdict = pickle.load(f)
            self.epsilon = readdict["epsilon"]
            self.training_data_mean = readdict["training_data_mean"]
            self.training_data_std = readdict["training_data_std"]
            self.training_data_min = readdict["training_data_min"]
            self.training_data_max = readdict["training_data_max"]
        else:
            self.training_data = training_data
            self.epsilon = 1e-100
            self.training_data_mean = np.mean(training_data, axis = 0)
            self.training_data_std = np.std(training_data, axis = 0) + self.epsilon
            self.training_data_min = np.min(training_data, axis = 0)
            self.training_data_max = np.max(training_data, axis = 0)
    def save(self, path):
        with open(path, 'wb') as f:
            savedict = {"epsilon":self.epsilon, "training_data_mean":self.training_data_mean, "training_data_std":self.training_data_std, 
                        "training_data_min":self.training_data_min, "training_data_max":self.training_data_max}
            pickle.dump(savedict, f)
        
    def fit(self, data, mode = "normalization"):
        if mode == "normalization":
            transform = (data - self.training_data_min) / (self.training_data_max - self.training_data_min)
            return transform
        elif mode == "standardization":
            transform = (data - self.training_data_mean) / self.training_data_std
            return transform

def evaluation(outputs, labels):
    outputs[outputs>=0.5] = 1
    outputs[outputs<0.5] = 0
    correct = torch.sum(torch.eq(outputs, labels)).item()
    return correct

def evalution_outputs_transform(outputs, criteria = 0.5):
    outputs[outputs>=criteria] = 1
    outputs[outputs<criteria] = 0
    return outputs

def train_test_plot(data, order = ["Train", "Test"], mode = "Acc", saving_name = "./acc_plot.png"):
    for idx, one_data in enumerate(data):
        plt.plot(one_data, label = order[idx])
    plt.xlabel("Epoch")
    if mode == "Acc":
        plt.ylabel("Accuracy")
        plt.title("Accuracy")
    elif mode == "Loss":
        plt.ylabel("Loss")
        plt.title("Loss")
    elif mode == "F1":
        plt.ylabel("F1")
        plt.title("F1 score")
    plt.legend()
    plt.savefig(saving_name)
    #plt.show()