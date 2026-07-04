import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np

# ============================
#  Dataset
# ============================
class TrainDataset(Dataset):
    def __init__(self, X_train_sys, X_train_bgr,X_train_type,seq_lengths_train):
        self.X_train_sys = torch.tensor(X_train_sys, dtype=torch.float32)
        self.X_train_bgr = torch.tensor(X_train_bgr, dtype=torch.float32)
        # self.ex_seq_lengths = torch.tensor(ex_seq_lengths_train, dtype=torch.long)
        self.seq_lengths = torch.tensor(seq_lengths_train, dtype=torch.long)
        self.X_train_type= torch.tensor(X_train_type, dtype=torch.long)

    def __len__(self):
        return len(self.X_train_sys)

    def __getitem__(self, idx):
        return self.X_train_sys[idx], self.X_train_bgr[idx], self.X_train_type[idx],self.seq_lengths[idx]

class ValDataset(Dataset):
    def __init__(self, X_val_sys, X_val_bgr,X_val_type, seq_lengths_val):
        self.X_val_sys = torch.tensor(X_val_sys, dtype=torch.float32)
        self.X_val_bgr = torch.tensor(X_val_bgr, dtype=torch.float32)
        # self.ex_seq_lengths = torch.tensor(ex_seq_lengths_val, dtype=torch.long)
        self.seq_lengths = torch.tensor(seq_lengths_val, dtype=torch.long)
        self.X_val_type = torch.tensor(X_val_type, dtype=torch.long)

    def __len__(self):
        return len(self.X_val_sys)

    def __getitem__(self, idx):
        return self.X_val_sys[idx], self.X_val_bgr[idx], self.X_val_type[idx],self.seq_lengths[idx]