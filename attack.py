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

class Attacker:
    def __init__(self, model, all_JS_data, all_JS_label, all_JS_length, all_JS_index, device):
        self.device = device
        self.data_len = len(all_JS_data)
        self.model = model
        self.model.to(device)
        self.model.eval()
        self.dataset = JSDataset(X=all_JS_data, y=all_JS_label, length=all_JS_length, ID=all_JS_index)
        self.loader = DataLoader(self.dataset, batch_size = 1, shuffle = False, num_workers = 1)
        print("dataloader", len(self.loader))

    # FGSM
    def fgsm_attack(self, JS, epsilon, data_grad):
        
        # find direction of gradient
        sign_data_grad = data_grad.sign()
        
        # add noise "epilon * direction" to the ori image
        perturbed_JS = JS + epsilon * sign_data_grad
        
        return perturbed_JS
    
    # PGD
    def pgd_attack(self, JS, ori_JS, eps, alpha, data_grad) :
        
        adv_JS = JS + alpha * data_grad.sign()
        eta = torch.clamp(adv_JS - ori_JS.data, min=-eps, max=eps)
        JS = ori_JS + eta

        return JS
    
    def attack(self, epsilon, alpha=0, attack_type = "FGSM"):
        adv_all = []
        wrong, fail, success = 0, 0, 0
        criterion = nn.BCELoss()
        for now, (data, label, length, ID) in enumerate(self.loader):
            #print(len(data), label, length, ID)
            print(str(now) + "|" +str(len(self.loader)), end="\r")
            
            target = label
            data, target = data.to(self.device).float(), target.to(self.device).float()
            data_raw = data
            
            # initial prediction
            output = self.model(data)
            init_pred = evalution_outputs_transform(output)

            # DO NOT ATTACK if incorrectly-classified
            if init_pred.item() != target.item():
                wrong += 1
                adv_all.append({"status":"wrong", "data":data.squeeze().detach().cpu().numpy(), "adv_data":data.squeeze().detach().cpu().numpy(), "label":label, "length":length, "ID":ID})
                continue
                
            # ATTACK if correctly-classified
            ############ ATTACK GENERATION ##############
            if attack_type == "FGSM":
                data.requires_grad = True
                output = self.model(data)
                loss = criterion(output, target)
                self.model.zero_grad()
                loss.backward()
                data_grad = data.grad.data
                perturbed_data = self.fgsm_attack(data, epsilon, data_grad)
            
            elif attack_type == "PGD":
                for i in range(50):
                    data.requires_grad = True
                    output = self.model(data)
                    loss = criterion(output, target)
                    self.model.zero_grad()
                    loss.backward()
                    data_grad = data.grad.data
                    
                    data = self.pgd_attack(data, data_raw, epsilon, alpha, data_grad).detach_()
                perturbed_data = data
            ############ ATTACK GENERATION ##############

            # prediction of adversarial image        
            output = self.model(perturbed_data)
            final_pred = evalution_outputs_transform(output)
            
            # if still correctly-predicting, attack failed
            if final_pred.item() == target.item():
                fail += 1
                adv_all.append({"status":"fail", "data":data_raw.squeeze().detach().cpu().numpy(), "adv_data":perturbed_data.squeeze().detach().cpu().numpy() , "label":label, "length":length, "ID":ID})
            
            # incorrectly-predicting, attack successfully
            else:
                success += 1
                adv_all.append({"status":"success", "data":data_raw.squeeze().detach().cpu().numpy(), "adv_data":perturbed_data.squeeze().detach().cpu().numpy() , "label":label, "length":length, "ID":ID})
        
        # calculate final accuracy 
        final_acc = (fail / (wrong + success + fail))
        noraml_acc = 1 - (wrong / (wrong + success + fail))
        print(f"Normal ACC = {noraml_acc}")
        print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, fail, len(self.loader), final_acc))
        print("Wrong: {} \tFail: {}\tSuccess: {}".format(wrong, fail, success))
        
        if attack_type == "FGSM":
            saving_name = f"ADV_{attack_type}_{epsilon}.pkl"
        if attack_type == "PGD":
            saving_name = f"ADV_{attack_type}_{epsilon}_{alpha}.pkl"
        
        with open("./Adv_example/"+ saving_name, 'wb') as f:
            pickle.dump(adv_all, f)
        return noraml_acc, final_acc
    
def attack_model(model_path, attacker_path, device = "cpu", adv_training = False):
    with open(attacker_path, 'rb') as f:
        adv_examples_data = pickle.load(f)
    adv_status_all = []
    adv_ori_data_all = []
    adv_adv_data_all = []
    adv_label_all = []
    adv_length_all = []
    adv_ID_all = []
    adv_mal_all = []
    if adv_training:
        adv_examples_data = adv_examples_data[8000:]
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
            
    test_dataset = JSDataset(X=adv_mal_all, y=adv_label_all, length=adv_length_all, ID=adv_ID_all)
    test_loader = DataLoader(test_dataset, batch_size = 64, shuffle = False, num_workers = 16)
    
    ori_dataset = JSDataset(X=adv_ori_data_all, y=adv_label_all, length=adv_length_all, ID=adv_ID_all)
    ori_loader = DataLoader(ori_dataset, batch_size = 64, shuffle = False, num_workers = 16)
    
    model = CountsNet().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    with torch.no_grad():
        test_pred_epoch = []
        test_ture_epoch = []
        test_total_acc = 0
        for i, (inputs, labels,_,_) in enumerate(test_loader):
            inputs = inputs.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.float)
            outputs = model(inputs)
            outputs[outputs != outputs] = 0
            outputs = outputs.squeeze()

            pred = evalution_outputs_transform(outputs)
            test_pred_epoch += list(pred)
            test_ture_epoch += list(labels)
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
        #####################################################
        ori_pred_epoch = []
        ori_ture_epoch = []
        ori_total_acc = 0
        for i, (inputs, labels,_,_) in enumerate(ori_loader):
            inputs = inputs.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.float)
            outputs = model(inputs)
            outputs[outputs != outputs] = 0
            outputs = outputs.squeeze()

            pred = evalution_outputs_transform(outputs)
            ori_pred_epoch += list(pred)
            ori_ture_epoch += list(labels)
        ori_pred_epoch = np.array(ori_pred_epoch)
        ori_ture_epoch = np.array(ori_ture_epoch)
        ori_TP = sum((ori_pred_epoch == 1) & (ori_ture_epoch == 1))
        ori_TN = sum((ori_pred_epoch == 0) & (ori_ture_epoch == 0))
        ori_FP = sum((ori_pred_epoch == 1) & (ori_ture_epoch == 0))
        ori_FN = sum((ori_pred_epoch == 0) & (ori_ture_epoch == 1))
        ori_ACC = (ori_TP + ori_TN)/(ori_TP+ori_TN+ori_FP+ori_FN)
        ori_precision = (ori_TP)/(ori_TP+ori_FP+0.0000000001)
        ori_recall = ori_TP/(ori_TP+ori_FN+0.0000000001)
        ori_F1 = 2/(1/ori_precision + 1/ori_recall)


        print(f'Ori | Acc: {ori_ACC*100:.3f} F1: {ori_F1:.3f}|| Adv Acc: {test_ACC*100:.3f} F1: {test_F1:.3f}')
    return ori_ACC, ori_F1, test_ACC, test_F1

def testing_model(model_path, device = "cpu", mode = "good"):
    with open('./good_counts_data.pkl', 'rb') as f:
        good_counts, good_data_idx, good_data_length = pickle.load(f)
    with open('./mal_counts_data.pkl', 'rb') as f:    
        mal_counts, mal_data_idx, mal_data_length = pickle.load(f)
    if mode == "good":
        all_JS_data = good_counts
        all_JS_label = np.zeros(len(good_counts))
        all_JS_length = good_data_length
        all_JS_index = good_data_idx
    elif mode == "mal":
        all_JS_data = mal_counts
        all_JS_label = np.ones(len(mal_counts))
        all_JS_length = mal_data_length
        all_JS_index = mal_data_idx
    elif mode == "all":
        all_JS_data = np.vstack((good_counts,mal_counts))
        all_JS_label = np.hstack((np.zeros(len(good_counts)),np.ones(len(mal_counts))))
        all_JS_length = np.hstack((good_data_length,mal_data_length))
        all_JS_index = np.hstack((good_data_idx, mal_data_idx))
    else:
        return
            
    test_dataset = JSDataset(X=all_JS_data, y=all_JS_label, length=all_JS_length, ID=all_JS_index)
    test_loader = DataLoader(test_dataset, batch_size = 64, shuffle = False, num_workers = 16)
    
    model = CountsNet().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    with torch.no_grad():
        test_pred_epoch = []
        test_ture_epoch = []
        test_total_acc = 0
        for i, (inputs, labels,_,_) in enumerate(test_loader):
            inputs = inputs.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.float)
            outputs = model(inputs)
            #outputs[outputs != outputs] = 0
            outputs = outputs.squeeze()

            pred = evalution_outputs_transform(outputs)
            test_pred_epoch += list(pred)
            test_ture_epoch += list(labels)
        test_pred_epoch = np.array(test_pred_epoch)
        test_ture_epoch = np.array(test_ture_epoch)
        test_TP = sum((test_pred_epoch == 1) & (test_ture_epoch == 1))
        test_TN = sum((test_pred_epoch == 0) & (test_ture_epoch == 0))
        test_FP = sum((test_pred_epoch == 1) & (test_ture_epoch == 0))
        test_FN = sum((test_pred_epoch == 0) & (test_ture_epoch == 1))
        print(test_TP,test_TN,test_FP,test_FN)
        test_ACC = (test_TP + test_TN)/(test_TP+test_TN+test_FP+test_FN)
        test_precision = (test_TP)/(test_TP+test_FP+0.0000000001)
        test_recall = test_TP/(test_TP+test_FN+0.0000000001)
        test_F1 = 2/(1/test_precision + 1/test_recall)

        print(f'{mode} Acc: {test_ACC*100:.3f} F1: {test_F1:.3f}')
    return test_ACC, test_F1