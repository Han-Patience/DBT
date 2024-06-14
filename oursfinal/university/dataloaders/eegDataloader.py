# from PyEMD import EMD
import numpy as np

import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
pd.options.display.max_columns = None  #列数
pd.options.display.max_rows = None     #行数

class Dataset_eeg(Dataset):
    def __init__(self, flag='train', scale=False) -> None:
        assert flag in ['train', 'val'], 'not implement!'
        self.flag = flag
        ann = pd.read_csv("/mnt/oursfinal/datasets/eeg2/{}_ann.txt".format(self.flag),
                          sep=',',
                          header=None,low_memory=False,on_bad_lines='skip')

        self.data = ann.values

    def __getitem__(self, index: int):
        val = self.data[index]
        label = int(val[0])
#         print(label)
        seq = val[1:]
#------------------------------------------------------------------EMD test
        
#         x=seq
#         emd = EMD()
#         imfs = emd.emd(x)
#         imf_imf=imfs[3]
        
#         seq = imf_imf
#------------------------------------------------------------------  
#         print(len(seq))
        return torch.tensor(label, dtype=torch.long), torch.tensor(seq, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.data)


def getDataLoader(*, batch_size):

    train_dataset = Dataset_eeg(flag='train')
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = Dataset_eeg(flag='val')
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)

    return train_dataloader, val_dataloader


if __name__ == '__main__':
    train_dataset = Dataset_eeg(flag='train')
    print(train_dataset[0])
    # print(len(set(train_dataset[0][0].numpy())))
