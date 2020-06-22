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

if __name__ == '__main__':
    with open('./good_counts_data.pkl', 'rb') as f:
        good_counts, good_data_idx, good_data_length = pickle.load(f)
    with open('./mal_counts_data.pkl', 'rb') as f:    
        mal_counts, mal_data_idx, mal_data_length = pickle.load(f)

    all_JS_data = np.vstack((good_counts,mal_counts))
    all_JS_label = np.hstack((np.zeros(len(good_counts)),np.ones(len(mal_counts))))
    all_JS_length = np.hstack((good_data_length,mal_data_length))
    all_JS_index = np.hstack((good_data_idx, mal_data_idx))

    X_train, X_test, y_train, y_test, L_train, L_test, ID_train, ID_test = train_test_split(all_JS_data, all_JS_label, all_JS_length, all_JS_index, test_size=0.25, random_state=42)
    print(f"Training data: {X_train.shape}")
    print(f"Testing data: {X_test.shape}")
    scaledata = ScaleData(X_train)
    X_train_scale = scaledata.fit(X_train, mode = "standardization")
    X_test_scale = scaledata.fit(X_test, mode = "standardization")
    device = torch.device("cpu") #"cuda:0" if torch.cuda.is_available() else 
    batch_size = 512
    n_epoch = 50
    lr = 0.01

    model = CountsNet().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr,momentum=0.9)

    train_dataset = JSDataset(X=X_train_scale, y=y_train, length=L_train, ID=ID_train)
    test_dataset = JSDataset(X=X_test_scale, y=y_test, length=L_test, ID=ID_test)
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, num_workers = 16)
    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = True, num_workers = 16)


    total_loss, total_acc, best_acc = 0, 0, 0
    train_acc_list = []
    train_loss_list = []
    train_F1_list = []
    test_acc_list = []
    test_loss_list = []
    test_F1_list = []

    for epoch in range(n_epoch):
        total_loss, total_acc = 0, 0
        model.train()
        train_pred_epoch = []
        train_ture_epoch = []
        test_pred_epoch = []
        test_ture_epoch = []

        for i, (inputs, labels,_,_) in enumerate(train_loader):
            print(f"process {i+1} / {len(train_loader)}", end="\r")
            inputs = inputs.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.float)
            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = outputs.squeeze()
            loss = criterion(outputs, labels) 
            loss.backward() 
            optimizer.step() 
            pred = evalution_outputs_transform(outputs)
            train_pred_epoch += list(pred)
            train_ture_epoch += list(labels)
            total_loss += loss.item()

        train_pred_epoch = np.array(train_pred_epoch)
        train_ture_epoch = np.array(train_ture_epoch)
        train_TP = sum((train_pred_epoch == 1) & (train_ture_epoch == 1))
        train_TN = sum((train_pred_epoch == 0) & (train_ture_epoch == 0))
        train_FP = sum((train_pred_epoch == 1) & (train_ture_epoch == 0))
        train_FN = sum((train_pred_epoch == 0) & (train_ture_epoch == 1))
        train_ACC = (train_TP + train_TN)/(train_TP+train_TN+train_FP+train_FN)
        train_precision = (train_TP)/(train_TP+train_FP)
        train_recall = train_TP/(train_TP+train_FN)
        train_F1 = 2/(1/train_precision + 1/train_recall)

        train_acc_list.append(train_ACC)
        train_loss_list.append(total_loss/len(train_loader))
        train_F1_list.append(train_F1)


        model.eval()
        with torch.no_grad():
            total_loss, total_acc = 0, 0
            for i, (inputs, labels,_,_) in enumerate(test_loader):
                inputs = inputs.to(device, dtype=torch.float)
                labels = labels.to(device, dtype=torch.float)
                outputs = model(inputs)
                #outputs = torch.clamp(outputs, 0, 1)
                outputs[outputs != outputs] = 0
                outputs = outputs.squeeze()
                loss = criterion(outputs, labels)
                pred = evalution_outputs_transform(outputs)
                test_pred_epoch += list(pred)
                test_ture_epoch += list(labels)
                total_loss += loss.item()
            test_pred_epoch = np.array(test_pred_epoch)
            test_ture_epoch = np.array(test_ture_epoch)
            test_TP = sum((test_pred_epoch == 1) & (test_ture_epoch == 1))
            test_TN = sum((test_pred_epoch == 0) & (test_ture_epoch == 0))
            test_FP = sum((test_pred_epoch == 1) & (test_ture_epoch == 0))
            test_FN = sum((test_pred_epoch == 0) & (test_ture_epoch == 1))
            test_ACC = (test_TP + test_TN)/(test_TP+test_TN+test_FP+test_FN)
            test_precision = (test_TP)/(test_TP+test_FP)
            test_recall = test_TP/(test_TP+test_FN)
            test_F1 = 2/(1/test_precision + 1/test_recall)

            test_acc_list.append(test_ACC)
            test_loss_list.append(total_loss/len(test_loader))
            test_F1_list.append(test_F1)
            print(f'Epoch {epoch+1} || Train | Loss:{total_loss/len(train_loader):.5f} Acc: {train_ACC*100:.3f} F1: {train_F1:.3f} || Valid | Loss:{total_loss/len(test_loader):.5f} Acc: {test_ACC*100:.3f} F1: {test_F1:.3f}')

    train_test_plot([train_acc_list, test_acc_list], mode = "Acc", saving_name = "./BASIC_part_ACC.png")
    train_test_plot([train_loss_list, test_loss_list], mode = "Loss", saving_name = "./BASIC_part_LOSS.png")
    train_test_plot([train_F1_list, test_F1_list], mode = "F1", saving_name = "./BASIC_part_F1.png")

    #train by all data
    print(f"ALL Training data: {all_JS_data.shape}")
    scaledata = ScaleData(all_JS_data)
    X_all_JS_scale = scaledata.fit(all_JS_data, mode = "standardization")

    device = torch.device("cpu") #"cuda:0" if torch.cuda.is_available() else 
    batch_size = 512
    n_epoch = 100
    lr = 0.01

    model = CountsNet().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr,momentum=0.9)

    train_dataset = JSDataset(X=X_all_JS_scale, y=all_JS_label, length=all_JS_length, ID=all_JS_index)
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, num_workers = 16)
    test_dataset = JSDataset(X=X_all_JS_scale, y=all_JS_label, length=all_JS_length, ID=all_JS_index)
    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = True, num_workers = 16)

    total_loss, total_acc, best_acc = 0, 0, 0
    train_acc_list = []
    train_loss_list = []
    train_F1_list = []
    test_acc_list = []
    test_loss_list = []
    test_F1_list = []

    for epoch in range(n_epoch):
        total_loss, total_acc = 0, 0
        model.train()
        train_pred_epoch = []
        train_ture_epoch = []
        test_pred_epoch = []
        test_ture_epoch = []

        for i, (inputs, labels,_,_) in enumerate(train_loader):
            print(f"process {i+1} / {len(train_loader)}", end="\r")
            inputs = inputs.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.float)
            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = outputs.squeeze()
            loss = criterion(outputs, labels) 
            loss.backward() 
            optimizer.step() 
            pred = evalution_outputs_transform(outputs)
            train_pred_epoch += list(pred)
            train_ture_epoch += list(labels)
            total_loss += loss.item()

        train_pred_epoch = np.array(train_pred_epoch)
        train_ture_epoch = np.array(train_ture_epoch)
        train_TP = sum((train_pred_epoch == 1) & (train_ture_epoch == 1))
        train_TN = sum((train_pred_epoch == 0) & (train_ture_epoch == 0))
        train_FP = sum((train_pred_epoch == 1) & (train_ture_epoch == 0))
        train_FN = sum((train_pred_epoch == 0) & (train_ture_epoch == 1))
        train_ACC = (train_TP + train_TN)/(train_TP+train_TN+train_FP+train_FN)
        train_precision = (train_TP)/(train_TP+train_FP)
        train_recall = train_TP/(train_TP+train_FN)
        train_F1 = 2/(1/train_precision + 1/train_recall)

        train_acc_list.append(train_ACC)
        train_loss_list.append(total_loss/len(train_loader))
        train_F1_list.append(train_F1)


        model.eval()
        with torch.no_grad():
            total_loss, total_acc = 0, 0
            for i, (inputs, labels,_,_) in enumerate(test_loader):
                inputs = inputs.to(device, dtype=torch.float)
                labels = labels.to(device, dtype=torch.float)
                outputs = model(inputs)
                #outputs = torch.clamp(outputs, 0, 1)
                outputs[outputs != outputs] = 0
                outputs = outputs.squeeze()
                loss = criterion(outputs, labels)
                pred = evalution_outputs_transform(outputs)
                test_pred_epoch += list(pred)
                test_ture_epoch += list(labels)
                total_loss += loss.item()
            test_pred_epoch = np.array(test_pred_epoch)
            test_ture_epoch = np.array(test_ture_epoch)
            test_TP = sum((test_pred_epoch == 1) & (test_ture_epoch == 1))
            test_TN = sum((test_pred_epoch == 0) & (test_ture_epoch == 0))
            test_FP = sum((test_pred_epoch == 1) & (test_ture_epoch == 0))
            test_FN = sum((test_pred_epoch == 0) & (test_ture_epoch == 1))
            test_ACC = (test_TP + test_TN)/(test_TP+test_TN+test_FP+test_FN)
            test_precision = (test_TP)/(test_TP+test_FP)
            test_recall = test_TP/(test_TP+test_FN)
            test_F1 = 2/(1/test_precision + 1/test_recall)

            test_acc_list.append(test_ACC)
            test_loss_list.append(total_loss/len(test_loader))
            test_F1_list.append(test_F1)
            print(f'Epoch {epoch+1} || Train | Loss:{total_loss/len(train_loader):.5f} Acc: {train_ACC*100:.3f} F1: {train_F1:.3f} || Valid | Loss:{total_loss/len(test_loader):.5f} Acc: {test_ACC*100:.3f} F1: {test_F1:.3f}')

        if (epoch + 1) % 10 == 0:
            model_name = f"CountModel_epoch_{epoch + 1}_{test_ACC}_{test_F1}.pkl"
            torch.save(model.state_dict(), "./Model/"+model_name)

    train_test_plot([train_acc_list, test_acc_list], mode = "Acc", saving_name = "./BASIC_ALL_ACC.png")
    train_test_plot([train_loss_list, test_loss_list], mode = "Loss", saving_name = "./BASIC_ALL_ACC.png")
    train_test_plot([train_F1_list, test_F1_list], mode = "F1", saving_name = "./BASIC_ALL_ACC.png")