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

def train_defense(train_data_list, test_data_list, batch_size =512, n_epoch = 50, lr=0.01, L1_regularization = 0, device="cuda:0", save_name = None):

    model = CountsNet().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    train_dataset = JSDataset(X=train_data_list[0], y=train_data_list[1], length=train_data_list[2], ID=train_data_list[3])
    test_dataset = JSDataset(X=test_data_list[0], y=test_data_list[1], length=test_data_list[2], ID=test_data_list[3])
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, num_workers = 16)
    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False, num_workers = 16)

    #total_loss, total_acc, best_acc = 0, 0, 0
    train_acc_list = []
    train_loss_list = []
    train_F1_list = []
    test_acc_list = []
    test_loss_list = []
    test_F1_list = []

    for epoch in range(n_epoch):
        train_total_loss, train_total_acc = 0, 0
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
            
            
            if L1_regularization > 0:
                regularization_loss = 0
                for param in model.parameters():
                    regularization_loss += torch.sum(torch.abs(param))
                loss = criterion(outputs, labels) + L1_regularization * regularization_loss
            else:
                loss = criterion(outputs, labels)
                
            loss.backward() 
            optimizer.step() 
            pred = evalution_outputs_transform(outputs)
            train_pred_epoch += list(pred)
            train_ture_epoch += list(labels)
            train_total_loss += loss.item()

        train_pred_epoch = np.array(train_pred_epoch)
        train_ture_epoch = np.array(train_ture_epoch)
        train_TP = sum((train_pred_epoch == 1) & (train_ture_epoch == 1))
        train_TN = sum((train_pred_epoch == 0) & (train_ture_epoch == 0))
        train_FP = sum((train_pred_epoch == 1) & (train_ture_epoch == 0))
        train_FN = sum((train_pred_epoch == 0) & (train_ture_epoch == 1))
        train_ACC = (train_TP + train_TN)/(train_TP+train_TN+train_FP+train_FN)
        train_precision = (train_TP)/(train_TP+train_FP+0.0000000001)
        train_recall = train_TP/(train_TP+train_FN+0.0000000001)
        train_F1 = 2/(1/train_precision + 1/train_recall)

        train_acc_list.append(train_ACC)
        train_loss_list.append(train_total_loss/len(train_loader))
        train_F1_list.append(train_F1)


        model.eval()
        with torch.no_grad():
            test_total_loss, test_total_acc = 0, 0
            for i, (inputs, labels,_,_) in enumerate(test_loader):
                inputs = inputs.to(device, dtype=torch.float)
                labels = labels.to(device, dtype=torch.float)
                outputs = model(inputs)
                #outputs = torch.clamp(outputs, 0, 1)
                outputs[outputs != outputs] = 0
                outputs = outputs.squeeze()
                
                if L1_regularization > 0:
                    regularization_loss = 0
                    for param in model.parameters():
                        regularization_loss += torch.sum(torch.abs(param))
                    loss = criterion(outputs, labels) + L1_regularization * regularization_loss
                else:
                    loss = criterion(outputs, labels)
                
                pred = evalution_outputs_transform(outputs)
                test_pred_epoch += list(pred)
                test_ture_epoch += list(labels)
                test_total_loss += loss.item()
            test_pred_epoch = np.array(test_pred_epoch)
            test_ture_epoch = np.array(test_ture_epoch)
            test_TP = sum((test_pred_epoch == 1) & (test_ture_epoch == 1))
            test_TN = sum((test_pred_epoch == 0) & (test_ture_epoch == 0))
            test_FP = sum((test_pred_epoch == 1) & (test_ture_epoch == 0))
            test_FN = sum((test_pred_epoch == 0) & (test_ture_epoch == 1))
            test_ACC = (test_TP + test_TN)/(test_TP+test_TN+test_FP+test_FN)
            test_precision = (test_TP)/(test_TP+test_FP+0.0000000001)
            test_recall = test_TP/(test_TP+test_FN+0.0000000001)
            test_F1 = 2/(1/test_precision + 1/test_recall)

            test_acc_list.append(test_ACC)
            test_loss_list.append(test_total_loss/len(test_loader))
            test_F1_list.append(test_F1)
            print(f'Epoch {epoch+1} || Train | Loss:{train_total_loss/len(train_loader):.5f} Acc: {train_ACC*100:.3f} F1: {train_F1:.3f} || Valid | Loss:{test_total_loss/len(test_loader):.5f} Acc: {test_ACC*100:.3f} F1: {test_F1:.3f}')
    if save_name != None:
        torch.save(model.state_dict(), "./Model/"+save_name)
    train_test_plot([train_acc_list, test_acc_list], mode = "Acc", saving_name = save_name+"_ACC.png")
    train_test_plot([train_loss_list, test_loss_list], mode = "Loss", saving_name = save_name+"_LOSS.png")
    train_test_plot([train_F1_list, test_F1_list], mode = "F1", saving_name = save_name+"_F1.png")