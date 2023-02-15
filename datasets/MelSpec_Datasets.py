# Create dataloader from pre created mel specs
import glob
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch

# Dataset for Entropy based Model

class MelSpecDataset(Dataset):

    def __init__(self,path):
        self.data_path = path
        file_list = glob.glob(self.data_path + "*")
        #print(file_list[:2])
        self.data = []
        for class_path in file_list:
            class_name = class_path.split("/")[-1].split('.')[0].split('_')[-1]
            self.data.append([class_path, class_name])
        #print(self.data)
        
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        file_path, class_name = self.data[idx]
        with open(file_path, 'rb') as f:
            arr = np.load(f)
        file_name = file_path.split('/')[-1]
        file_ = torch.from_numpy(arr)
        file_ = np.expand_dims(file_,axis=0)
        class_id = torch.tensor(int(class_name))
        return file_, class_id, file_name
