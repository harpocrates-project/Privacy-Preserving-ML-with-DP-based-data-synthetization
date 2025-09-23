"""
utility class to concert a csv data to a torch dataset
need to implement __len__ and __getitem__ function
"""

import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import torch

class CmdDataset(Dataset):
    def __init__(self, x, y) -> None:
        """
        input:
            path    path to data
            hb_type type of hearbeat, A for anomaly else will be labelled as normal data N
        """
        super().__init__()
        self.df = x.tolist()
        self.y = y.tolist()
       
    def __len__(self):
        """
        returns length of data set
        """
        return len(self.df)
    
    def __getitem__(self, index):
        """
        returns element at index
        """
        seq, label = torch.tensor(self.df[index]).float(), self.y[index]

        return seq, label
