import random
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from colorama import Fore, Style
from torch.nn import utils
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from thop import profile

from dataloaders import magnetDataloader, salinasDataloader, ecgDataloader,eegDataloader
from model.DBT import DBT
from util.utils import (EarlyStopping, metric, plotfigure, inform)

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = True



class Args():
    def __init__(self) -> None:
        self.epochs = 800
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.head = 3
        self.depth = 3
        self.dim = 512  #512 最佳
        self.learning_rate = 0.01
        self.batch_size = 4  #512 for 弱磁、遥感、心电；4 for 脑电
        self.dropout = 0.2  #0.2目前最佳
        self.patience = 80
        self.bestScore = 0.
        self.recode = [0., 0., 0.]
        
        #DataBlock部分的参数，全False则为一阶差分堆叠，调整其中任意一个为True则视为在DB部分执行了对应的特征工程
        self.shift = 1 #一阶差分（单一特征堆叠）
        self.is_rawdata = False #原始时序数据（单一特征堆叠）
        self.is_SOD = False #二阶差分（单一特征堆叠）
        self.layerNorm = False #逐行归一化 （单一特征堆叠）
        self.is_ms = False #多尺度特征（单一特征堆叠）
        self.is_AVGi = False #跃变均值（单一特征堆叠）
        self.is_Dd = False #远距离依赖（单一特征堆叠）
        self.is_DBT = True #Data Block Transformer
                
        #更改数据集名称: 
#        self.data = 'magnet2'
#        self.data = 'salinas'
#        self.data = 'ecg'
        self.data = 'eeg'

        #无需修改的参数: 
        self.max_len = None
        self.num_class = None


def main():
    args = Args()

    data_type = {
        'magnet2': {
            'max_len': 501,
            'nums_class': 2,
            'function': magnetDataloader.getDataLoader  # magnet2.getDataLoader
        },
        'salinas': {
            'max_len': 204, 
            'nums_class': 17, 
            'function': salinasDataloader.getDataLoader
        },
        'ecg': {
            'max_len': 260, 
            'nums_class': 5, 
            'function': ecgDataloader.getDataLoader
        },
        'eeg': {
            'max_len': 4097, 
            'nums_class': 2, 
            'function': eegDataloader.getDataLoader
        },
    }
    args.max_len = data_type[args.data]['max_len']
    args.num_class = data_type[args.data]['nums_class']
    getDataLoader = data_type[args.data]['function']

    train_dataloader, valid_dataloader = getDataLoader(batch_size=args.batch_size)
    
    
    model = DBT(heads=args.head,
                    shift=args.shift,
                    layerNorm=args.layerNorm,
                    num_class=args.num_class,
                    max_len=args.max_len,
                    dropout=args.dropout,
                    emb_dropout=args.dropout,
                    dim=args.dim,
                    mlp_dim=args.dim,
                    isMs=args.is_ms,
                    isDBT=args.is_DBT,
                    israwdata=args.is_rawdata,
                    isAVGi=args.is_AVGi,
                    isSOD=args.is_SOD,
                    isDd=args.is_Dd).to(args.device).to(torch.float32)

    setting = "figs/" + \
            "data{}_head{}_depth{}_dim{}_sf{}_lr{}_ln{}_ms{}_rd{}_Dd{}_SOD{}_AVGi{}_DBT{}_bz{}_dpot{}_pt{}_md{}".format(
            args.data, args.head, args.depth,  args.dim, args.shift,
            args.learning_rate, args.layerNorm, args.is_ms, args.is_rawdata, args.is_Dd, args.is_SOD, args.is_AVGi, args.is_DBT, args.batch_size,
            args.dropout, args.patience,model._get_name())
    setting2 = "data_{}_model_{}".format(args.data, model._get_name())
    criterion = nn.CrossEntropyLoss()


    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.learning_rate,
                                momentum=0.9,
                                weight_decay=0.02,
                                dampening=0.618)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)  #, eps=1e-8)

    train_epochs_loss = []
    valid_epochs_loss = []
    train_acc = []
    val_acc = []
    early_stopping = EarlyStopping(verbose=True, patience=args.patience)

    for epoch in range(args.epochs):
        model.train()
        train_epoch_loss = []
        acc, nums = 0, 0
        for idx, (label, signal) in enumerate(tqdm(train_dataloader)):
            signal = signal.to(args.device)
#             print(signal.shape,label.shape)
            label = label.to(args.device)
#             print(label)
            outputs = model(signal)
#             print(outputs)
            optimizer.zero_grad()
#             label = label.to(torch.long)
            loss = criterion(outputs, label)
#             print(loss)
            loss.backward()
            optimizer.step()
            train_epoch_loss.append(loss.item())
            acc += sum(outputs.max(axis=1)[1] == label).cpu()
            nums += label.size()[0]
        train_epochs_loss.append(np.average(train_epoch_loss))
        train_acc.append(100 * acc / nums)
        print("train acc = {:.3f}%,loss = {}".format(100 * acc / nums, np.average(train_epoch_loss)))
        torch.cuda.empty_cache()
        # =========================val=========================
        with torch.no_grad():
            model.eval()
            val_epoch_loss = []
            acc, nums = 0, 0
            P, N, detect, false_alarm = 0, 0, 0, 0
            gt, pd = [], []
            for idx, (label, signal) in enumerate(tqdm(valid_dataloader)):
                signal = signal.to(args.device)  #.to(torch.float)
                label = label.to(args.device)
                
                outputs = model(signal)
#                 label = label.to(torch.long)
                loss = criterion(outputs, label)
                val_epoch_loss.append(loss.item())

                acc += sum(outputs.max(axis=1)[1] == label).cpu()
                nums += label.size()[0]

                elem1, elem2, elem3, elem4 = metric(outputs=outputs, label=label)
                P += elem1
                N += elem2
                detect += elem3
                false_alarm += elem4
                gt += label.tolist()
                pd += (outputs.max(axis=1)[1]).tolist()

            valid_epochs_loss.append(np.average(val_epoch_loss))
            val_acc.append(100 * acc / nums)

            if val_acc[-1] > args.bestScore:
                args.bestScore = val_acc[-1]
                args.recode = [100 * detect / P, 100. - 100 * detect / P, 100 * false_alarm / N]
                Gt, Pd = gt, pd

            print("epoch = {}, valid acc = {:.2f}%, loss = {}".format(epoch, 100 * acc / nums,
                                                                      np.average(val_epoch_loss)))

            print("detect = {:.2f}%, miss_rate = {:.2f}%, false_alarm = {:.2f}%".format(
                100 * detect / P, 100. - 100 * detect / P, 100 * false_alarm / N))

        # ==================early stopping=====================
        early_stopping(valid_epochs_loss[-1], model=model, path=setting + model._get_name())
        if early_stopping.early_stop:
            np.save('res/{}_{}_pred.npy'.format(setting2, args.bestScore),
                    np.array(Pd))
            np.save('res/{}_{}_gt.npy'.format(setting2, args.bestScore),
                    np.array(Gt))
            print("early_stopping!!!")
            inform(bestScore=args.bestScore, wechat=True)
            break
        # ==================adjust lr==========================
        if early_stopping.counter in [44, 60, 70, 256]:
            lr = args.learning_rate * 1.5
            args.learning_rate = lr
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
                print('{}Updating learning rate to = {}{}'.format(Fore.BLUE, lr, Style.RESET_ALL))

        print("=" * 36 + "lr = {} bestScore = {:.3f} another = {}".format(
            args.learning_rate, args.bestScore, list(map(lambda fc: round(fc, 2), args.recode))))

    plotfigure(train_acc, val_acc, train_epochs_loss, valid_epochs_loss, setting, args.bestScore, wechat=False)


if __name__ == "__main__":
    main()
