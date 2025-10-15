from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
def mask_data(data, ratio):
    data = data.to_numpy().copy()
    bools = np.random.uniform(0, 1, size=data.shape)
    data[bools < ratio] = -1
    return data

def to_categorical(y):
    codebook = np.array([[1,0,0], [0,1,0], [0,0,1], [0,0,0]])
    y = np.asarray(y, dtype=int)
    return codebook[y]

class GenotypeDataset(Dataset):
    def __init__(self, data, missing_perc=0.1):
        self.data = data
        self.missing_perc = missing_perc

    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        x = self.data[idx].copy()
        missing_size = int(self.missing_perc * len(x))
        missing_index = np.random.randint(len(x), size=missing_size)
        x[missing_index, :] = [0, 0, 0]
        y = self.data[idx]

        if len(x.shape)==2:
            x=x.transpose(1,0)
            y=y.transpose(1,0)

        if len(x.shape)==3:
            x=x.transpose(0,2,1)
            y=y.transpose(0,2,1)

        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


class GenotypeDataset2(Dataset):
    def __init__(self, data, y):
        self.data = np.asarray(data, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.float32)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.y[idx]
        if len(x.shape) == 2:
            x = x.transpose(1, 0)
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def process_data(genotype: np.array,
                 phenotype:np.array, 
                 test_split_ratio_per=0.1 ,
                 test_split_ratio_bv=0.1,
                 use_seed=False,
                 random_seed=42,
                 batch_size_pre=16,
                 batch_size_bv=16,
                 missing_perc=0.5,
                 to_split_in_pre=False
                 ):
    if use_seed:
        if isinstance(random_seed,int):
            pass
        else:
            raise ValueError("random_seed must be int")
    genotype_one_hot = to_categorical(genotype)
    
    if to_split_in_pre:
        if use_seed:
            train_dataset, valid_dataset = train_test_split(genotype_one_hot, test_size=test_split_ratio_per, random_state=random_seed)
        else:
            train_dataset, valid_dataset = train_test_split(genotype_one_hot, test_size=test_split_ratio_per)
        train_dataset = GenotypeDataset(train_dataset,missing_perc=missing_perc)
        valid_dataset = GenotypeDataset(valid_dataset,missing_perc=missing_perc)
        train_loader_pre = DataLoader(train_dataset ,batch_size=batch_size_pre, shuffle=True)
        valid_loader_pre = DataLoader(train_dataset ,batch_size=len(train_dataset), shuffle=True)
    else:
        train_dataset = GenotypeDataset(genotype_one_hot,missing_perc=missing_perc)
        train_loader_pre = DataLoader(train_dataset ,batch_size=batch_size_pre, shuffle=True)
        
    # train_dataset = GenotypeDataset(genotype_one_hot, missing_perc=missing_perc)
    # train_loader_pre = DataLoader(train_dataset ,batch_size=batch_size_pre, shuffle=True)
    # valid_loader_pre = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)
    if use_seed:
        X_train,X_test,y_train,y_test = train_test_split(genotype_one_hot, phenotype, test_size=test_split_ratio_bv, random_state=random_seed)
    else:
        X_train,X_test,y_train,y_test = train_test_split(genotype_one_hot, phenotype, test_size=test_split_ratio_bv)
    train_dataset= GenotypeDataset2(X_train,y_train)
    test_dataset= GenotypeDataset2(X_test,y_test)
    train_loader_bv = DataLoader(train_dataset, batch_size=batch_size_bv, shuffle=True)
    test_loader_bv = DataLoader(test_dataset,batch_size=len(test_dataset),shuffle=True)
    if to_split_in_pre:
        return train_loader_pre, valid_loader_pre, train_loader_bv, test_loader_bv
    else:
        return train_loader_pre, None, train_loader_bv, test_loader_bv
