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
from attack import *
from defense import *
from sklearn.model_selection import train_test_split

if __name__ == '__main__':

    model_saving_path = "./Model/"
    adv_example_saving_path = "./Adv_example/"

    all_models = os.listdir(model_saving_path)
    all_models.sort()
    all_attackers = os.listdir(adv_example_saving_path)
    all_attackers.sort()

    model_list = []
    all_adv_ACC_list = []
    all_adv_F1_list = []
    all_ori_ACC_list = []
    all_ori_F1_list = []
    for model_test in all_models:
        if "DEFENSEMODEL" in model_test:
            print("========================================")
            print(model_test)
            print("========================================")
            model_list.append(model_test)
            attacker_list = []
            all_adv_ACC_list_attacker = []
            all_adv_F1_list_attacker = []
            all_ori_ACC_list_attacker = []
            all_ori_F1_list_attacker = []
            for attacker in all_attackers:
                adv_example_path = os.path.join(adv_example_saving_path, attacker)
                model_path = os.path.join(model_saving_path, model_test)
                print(model_test, attacker)
                attacker_list.append(attacker)
                if "adv_training" in model_test:
                    ori_acc, ori_F1, at_acc, at_F1 = attack_model(model_path, adv_example_path, adv_training=True)
                else:
                    ori_acc, ori_F1, at_acc, at_F1 = attack_model(model_path, adv_example_path)
                ori_acc, ori_F1, at_acc, at_F1 = attack_model(model_path, adv_example_path)
                all_adv_ACC_list_attacker.append(at_acc)
                all_adv_F1_list_attacker.append(at_F1)
                all_ori_ACC_list_attacker.append(ori_acc)
                all_ori_F1_list.append(ori_F1)
            all_ori_F1_list_attacker = []
            all_adv_ACC_list.append(all_adv_ACC_list_attacker)
            all_adv_F1_list.append(all_adv_F1_list_attacker)
            all_ori_ACC_list.append(all_ori_ACC_list_attacker)
            all_ori_F1_list.append(all_ori_F1_list)

    plt.figure(figsize=(8, 6))
    for i, (model, ori, adv) in enumerate(zip(model_list, all_ori_ACC_list, all_adv_ACC_list)):
        if "L1" in model:
            if float(model[:-3].split("_")[-1]) <= 0.003:
                to_draw_list = []
                to_draw_list_x = []
                to_draw_list.append(all_ori_ACC_list[i][0])
                to_draw_list_x.append(0)
                for j, attacker_name in enumerate(attacker_list):
                    if "FGSM" in attacker_name:
                        x = float(attacker_name[:-4].split("_")[-1])
                        to_draw_list_x.append(x)
                        to_draw_list.append(all_adv_ACC_list[i][j])
                plt.plot(to_draw_list_x, to_draw_list, label = model[:-3].split("_")[-1],marker=".")
    plt.ylabel("Accuracy",size= 15)
    plt.xlabel("FGSM epsilon",size=15)
    plt.title("Accuracy of L1 regularization against FSGM attack",size= 15)
    plt.legend(title="L1 reg.")
    plt.save("./FGSM_attack_L1.png")


    plt.figure(figsize=(8, 6))
    for i, (model, ori, adv) in enumerate(zip(model_list, all_ori_ACC_list, all_adv_ACC_list)):
        if "L1" in model:
            if float(model[:-3].split("_")[-1]) <= 0.003:
                to_draw_list = []
                to_draw_list_x = []
                to_draw_list.append(all_ori_ACC_list[i][0])
                to_draw_list_x.append(0)
                for j, attacker_name in enumerate(attacker_list):
                    if "PGD" in attacker_name:
                        x = float(attacker_name[:-4].split("_")[-2])
                        to_draw_list_x.append(x)
                        to_draw_list.append(all_adv_ACC_list[i][j])
                to_draw_list_x[4], to_draw_list_x[5] = to_draw_list_x[5], to_draw_list_x[4]
                to_draw_list[4], to_draw_list[5] = to_draw_list[5], to_draw_list[4]
                plt.plot(to_draw_list_x, to_draw_list, label = model[:-3].split("_")[-1],marker=".")
    plt.ylabel("Accuracy",size= 15)
    plt.xlabel("PGD epsilon",size=15)
    plt.title("Accuracy of L1 regularization against PGD attack",size= 15)
    plt.legend(title="L1 reg.")
    plt.save("./PGD_attack_L1.png")


    plt.figure(figsize=(8, 6))
    for i, (model, ori, adv) in enumerate(zip(model_list, all_ori_ACC_list, all_adv_ACC_list)):
        if "adv_training_ADV_FGSM" in model:
            to_draw_list = []
            to_draw_list_x = []
            to_draw_list.append(all_ori_ACC_list[i][0])
            to_draw_list_x.append(0)
            for j, attacker_name in enumerate(attacker_list):
                if "FGSM" in attacker_name:
                    x = float(attacker_name[:-4].split("_")[-1])
                    to_draw_list_x.append(x)
                    to_draw_list.append(all_adv_ACC_list[i][j])
            #to_draw_list_x[4], to_draw_list_x[5] = to_draw_list_x[5], to_draw_list_x[4]
            #to_draw_list[4], to_draw_list[5] = to_draw_list[5], to_draw_list[4]
            plt.plot(to_draw_list_x, to_draw_list, label = model[:-7].split("_")[-1],marker=".")
    plt.ylabel("Accuracy",size= 15)
    plt.xlabel("FGSM epsilon",size=15)
    plt.title("Accuracy of FGSM Adv training against FGSM attack",size= 20)
    plt.legend(loc="lower right", title="FGSM epsilon.")
    plt.save("./FGSM_attack_FGSM.png")


    plt.figure(figsize=(8, 6))
    for i, (model, ori, adv) in enumerate(zip(model_list, all_ori_ACC_list, all_adv_ACC_list)):
        if "adv_training_ADV_FGSM" in model:
            to_draw_list = []
            to_draw_list_x = []
            to_draw_list.append(all_ori_ACC_list[i][0])
            to_draw_list_x.append(0)
            for j, attacker_name in enumerate(attacker_list):
                if "PGD" in attacker_name:
                    x = float(attacker_name[:-4].split("_")[-2])
                    to_draw_list_x.append(x)
                    to_draw_list.append(all_adv_ACC_list[i][j])
            to_draw_list_x[4], to_draw_list_x[5] = to_draw_list_x[5], to_draw_list_x[4]
            to_draw_list[4], to_draw_list[5] = to_draw_list[5], to_draw_list[4]
            plt.plot(to_draw_list_x, to_draw_list, label = model[:-7].split("_")[-1],marker=".")
    plt.ylabel("Accuracy",size= 15)
    plt.xlabel("PGD epsilon",size=15)
    plt.title("Accuracy of FGSM Adv training against PGD attack",size= 20)
    plt.legend(title="FGSM epsilon.")
    plt.save("./PGD_attack_FGSM.png")

    plt.figure(figsize=(8, 6))
    for i, (model, ori, adv) in enumerate(zip(model_list, all_ori_ACC_list, all_adv_ACC_list)):
        if "adv_training_ADV_PGD" in model:
            to_draw_list = []
            to_draw_list_x = []
            to_draw_list.append(all_ori_ACC_list[i][0])
            to_draw_list_x.append(0)
            for j, attacker_name in enumerate(attacker_list):
                if "FGSM" in attacker_name:
                    x = float(attacker_name[:-4].split("_")[-1])
                    to_draw_list_x.append(x)
                    to_draw_list.append(all_adv_ACC_list[i][j])
            #to_draw_list_x[4], to_draw_list_x[5] = to_draw_list_x[5], to_draw_list_x[4]
            #to_draw_list[4], to_draw_list[5] = to_draw_list[5], to_draw_list[4]
            plt.plot(to_draw_list_x, to_draw_list, label = model[:-7].split("_")[-2],marker=".")
    plt.ylabel("Accuracy",size= 15)
    plt.xlabel("FGSM epsilon",size=15)
    plt.title("Accuracy of PGD Adv training against FGSM attack",size= 20)
    plt.legend(loc="lower right", title="PGD epsilon.")
    plt.save("./FGSM_attack_PGD.png")

    plt.figure(figsize=(8, 6))
    for i, (model, ori, adv) in enumerate(zip(model_list, all_ori_ACC_list, all_adv_ACC_list)):
        if "adv_training_ADV_PGD" in model:
            to_draw_list = []
            to_draw_list_x = []
            to_draw_list.append(all_ori_ACC_list[i][0])
            to_draw_list_x.append(0)
            for j, attacker_name in enumerate(attacker_list):
                if "PGD" in attacker_name:
                    x = float(attacker_name[:-4].split("_")[-2])
                    to_draw_list_x.append(x)
                    to_draw_list.append(all_adv_ACC_list[i][j])
            to_draw_list_x[4], to_draw_list_x[5] = to_draw_list_x[5], to_draw_list_x[4]
            to_draw_list[4], to_draw_list[5] = to_draw_list[5], to_draw_list[4]
            plt.plot(to_draw_list_x, to_draw_list, label = model[:-7].split("_")[-2],marker=".")
    plt.ylabel("Accuracy",size= 15)
    plt.xlabel("PGD epsilon",size=15)
    plt.title("Accuracy of PGD Adv training against PGD attack",size= 20)
    plt.legend(title="PGD epsilon.")
    plt.save("./PGD_attack_PGD.png")


    plt.figure(figsize=(8, 6))
    for i, (model, ori, adv) in enumerate(zip(model_list, all_ori_ACC_list, all_adv_ACC_list)):
        if "gau_noise" in model:
            to_draw_list = []
            to_draw_list_x = []
            to_draw_list.append(all_ori_ACC_list[i][0])
            to_draw_list_x.append(0)
            for j, attacker_name in enumerate(attacker_list):
                if "FGSM" in attacker_name:
                    x = float(attacker_name[:-4].split("_")[-1])
                    to_draw_list_x.append(x)
                    to_draw_list.append(all_adv_ACC_list[i][j])
            plt.plot(to_draw_list_x, to_draw_list, label = model[:-3].split("_")[-1],marker=".")
    plt.ylabel("Accuracy",size= 15)
    plt.xlabel("FGSM epsilon",size=15)
    plt.title("Accuracy of Gau. noise against FSGM attack",size= 15)
    plt.legend(title="Gua. noise")
    plt.save("./FGSM_attack_GAU.png")

    plt.figure(figsize=(8, 6))
    for i, (model, ori, adv) in enumerate(zip(model_list, all_ori_ACC_list, all_adv_ACC_list)):
        if "gau_noise" in model:
            to_draw_list = []
            to_draw_list_x = []
            to_draw_list.append(all_ori_ACC_list[i][0])
            to_draw_list_x.append(0)
            for j, attacker_name in enumerate(attacker_list):
                if "PGD" in attacker_name:
                    x = float(attacker_name[:-4].split("_")[-2])
                    to_draw_list_x.append(x)
                    to_draw_list.append(all_adv_ACC_list[i][j])
            to_draw_list_x[4], to_draw_list_x[5] = to_draw_list_x[5], to_draw_list_x[4]
            to_draw_list[4], to_draw_list[5] = to_draw_list[5], to_draw_list[4]
            plt.plot(to_draw_list_x, to_draw_list, label = model[:-3].split("_")[-1],marker=".")
    plt.ylabel("Accuracy",size= 15)
    plt.xlabel("PGD epsilon",size=15)
    plt.title("Accuracy of Gau. noise against PGD attack",size= 15)
    plt.legend(title="Gau. noise")
    plt.save("./PGD_attack_GAU.png")