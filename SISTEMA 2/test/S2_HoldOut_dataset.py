import os
import torch
from torch.utils.data import Dataset
import pandas as pd


class HoldOutDataset(Dataset):
    def __init__(self, features_dir, df):
        self.features_dir = features_dir
        self.patient_ids = [f.replace(".pt", "") for f in os.listdir(features_dir)]
        self.df = df

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        pid = self.patient_ids[idx]

        features = torch.load(os.path.join(self.features_dir, pid + ".pt"), weights_only=True)

        # Get label
        label_str = self.df[self.df["CODI"] == pid]["DENSITAT"].values
        if len(label_str) == 0:
            raise ValueError(f"No label for patient {pid}")

        label = 0.0 if label_str[0] == "NEGATIVA" else 1.0

        return pid, features, torch.tensor([label], dtype=torch.float32)
