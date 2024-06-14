import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from colorama import Fore, Style
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
import time


class EarlyStopping():
    def __init__(self, patience=5, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        # print("val_loss={}".format(val_loss))
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'{Fore.RED}EarlyStopping counter: {self.counter} out of {self.patience}{Style.RESET_ALL}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(
                f'{Fore.GREEN}Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...{Style.RESET_ALL}'
            )
        # torch.save(model.state_dict(), path + '_checkpoint.pth')
        self.val_loss_min = val_loss



def metric(outputs, label):
    """[计算检测率和虚警值]

    Args:
        outputs ([type]): [神经网络输出]
        label ([type]): [ground truth]

    Returns:
        [P, N, detect, false_alarm]: [阳性，阴性，检测值，虚警值]
        [warning]: 是个数不是百分比
    """
    out = outputs.max(axis=1)[1]
    P = label.sum().item()
    detect = (out & label).sum().item()
    false_alarm = out.sum().item() - detect
    N = label.size()[0] - P

    return P, N, detect, false_alarm
    """[usage]
        P, N, detect, false_alarm = 0, 0, 0, 0
            elem1, elem2, elem3, elem4 = metric(outputs=outputs, label=label)
            P += elem1
            N += elem2
            detect += elem3
            false_alarm += elem4
        print("detect = {:.3f}%, miss_rate = {:.3f}%, false_alarm = {:.3f}%".format(
            100*detect/P, 100.-100*detect/P, 100*false_alarm/N))
    """
    """_summary_

    Returns:
        _type_: _description_
    """


def plotfigure(train_acc, val_acc, train_epochs_loss, valid_epochs_loss, setting, bestScore, wechat=False):
    import matplotlib.pyplot as plt
    import requests

    plt.figure(figsize=(12, 4))
    plt.subplot(121)
    plt.plot(train_acc[:], '-', label="train_acc")
    plt.plot(val_acc[:], '-', label="acc@1")
    plt.title("Accuracy")
    plt.legend()
    plt.subplot(122)
    plt.plot(train_epochs_loss[1:], '-', label="train_loss")
    plt.plot(valid_epochs_loss[1:], '-', label="valid_loss")
    plt.title("epochs_loss")
    plt.legend()
    plt.grid()
    plt.savefig(setting + str(bestScore) + ".png")
    # plt.show()

    if wechat:
        requests.get(
            "http://www.pushplus.plus/send?token=24af4bfe58114caebafb91a10cf2f1df&title=程序通知&content={}&template=html"
            .format(bestScore))


def inform(bestScore, wechat=False):
    import requests
    if wechat:
        requests.get(
            "http://www.pushplus.plus/send?token=24af4bfe58114caebafb91a10cf2f1df&title=程序通知&content={}&template=txt"
            .format(bestScore))

