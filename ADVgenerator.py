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
from argparse import ArgumentParser

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-M", "--model_path", help="the basic model to generate adv. examples", default = True, dest="model_path")
    parser.add_argument("-F", "--FGSM", help="generate FGSM adv example or not", default = True, dest="FGSM")
    parser.add_argument("-P", "--PGD", help="generate PGD adv example or not", default = True, dest="PGD")
    args = parser.parse_args()
    
    with open('./good_counts_data.pkl', 'rb') as f:
        good_counts, good_data_idx, good_data_length = pickle.load(f)
    with open('./mal_counts_data.pkl', 'rb') as f:    
        mal_counts, mal_data_idx, mal_data_length = pickle.load(f)
    mal_labels = np.ones(len(mal_counts))
    scaledata = ScaleData(read_from = "./Model/scale_paras.pkl")
    mal_counts_scale = scaledata.fit(mal_counts, mode = "standardization")
    device  = "cpu"
    model = CountsNet()

    model.load_state_dict(torch.load(args.model_path))
    attacker = Attacker(model, mal_counts_scale, mal_labels, mal_data_length, mal_data_idx, device=device)
    epsilons = [0.005, 0.01, 0.05, 0.1, 0.150, 0.2, 0.5]

    torch.multiprocessing.set_sharing_strategy('file_system')
    if args.FGSM:
        normal_acc_list_FGSM = []
        attack_acc_list_FGSM = []
        for eps in epsilons:
            print("===================================================")
            NORMAL_ACC, FINAL_ACC = attacker.attack(eps, alpha=1/500, attack_type = "FGSM")
            normal_acc_list_FGSM.append(NORMAL_ACC)
            attack_acc_list_FGSM.append(FINAL_ACC)
            print("===================================================")
    if args.PGD:
        normal_acc_list_PGD = []
        attack_acc_list_PGD = []
        epsilons = [0.005, 0.01, 0.05, 0.1, 0.150, 0.2, 0.5]
        for eps in epsilons:
            print("===================================================")
            NORMAL_ACC, FINAL_ACC = attacker.attack(eps, alpha=1/500, attack_type = "PGD")
            normal_acc_list_PGD.append(NORMAL_ACC)
            attack_acc_list_PGD.append(FINAL_ACC)
        print("===================================================")