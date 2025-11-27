# ---------- system2_dataset.py ----------
"""
Dataset and utilities for System2
Contains:
 - PatientFeatureDataset: loads per-patient .pt feature files -> (features [1,N,M], label, pat_id)
 - PatchTripletDataset: samples triplets using the annotations Excel, can use encoder_fn for on-the-fly features
"""

import os
import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
from torchvision import transforms
import random

class PatientFeatureDataset(Dataset):
    def __init__(self, features_dir, patient_list, patient_label_map, device='cpu'):
        self.features_dir = features_dir
        self.patient_list = patient_list
        self.patient_label_map = patient_label_map
        self.device = device

    def __len__(self):
        return len(self.patient_list)

    def __getitem__(self, idx):
        pat = self.patient_list[idx]
        path = os.path.join(self.features_dir, f"{pat}.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Features not found: {path}")
        feats = torch.load(path)  # [N_patches, feat_dim]
        feats = feats.float()
        feats = feats.unsqueeze(0)  # [1, N, M]
        label = int(self.patient_label_map.get(pat, 0))
        return feats, label, pat

class PatchTripletDataset(Dataset):
    def __init__(self, annotations_xlsx, annotated_base_dir, transform=None,
                 precomputed_patch_features_dir=None, encoder_fn=None, device='cpu'):
        df = pd.read_excel(annotations_xlsx, dtype=str)
        df = df.rename(columns=lambda s: s.strip())
        df = df[df['Presence'].isin([1, -1, '1', '-1'])].copy()
        df['Window_ID'] = df['Window_ID'].astype(str).str.zfill(5)
        df['Pat_Section'] = df['Pat_ID'].astype(str) + "_" + df['Section_ID'].astype(str)

        self.pos = df[df['Presence'].astype(int) == 1]
        self.neg = df[df['Presence'].astype(int) == -1]
        self.pos_list = list(self.pos[['Pat_Section', 'Window_ID']].itertuples(index=False, name=None))
        self.neg_list = list(self.neg[['Pat_Section', 'Window_ID']].itertuples(index=False, name=None))
        self.annotated_base_dir = annotated_base_dir
        self.precomputed_patch_features_dir = precomputed_patch_features_dir
        self.encoder_fn = encoder_fn
        self.transform = transform or transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])
        self.device = device

    def __len__(self):
        # large sample size; sampling random triplets on the fly
        return max(100000, len(self.pos_list)*2)

    def _load_patch_feature(self, patsec, window_id):
        if self.precomputed_patch_features_dir:
            fn = os.path.join(self.precomputed_patch_features_dir, f"{patsec}_{window_id}.pt")
            if os.path.exists(fn):
                return torch.load(fn).float().to(self.device)
        img_path = os.path.join(self.annotated_base_dir, patsec, f"{window_id}.png")
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Patch image not found: {img_path}")
        img = Image.open(img_path).convert('RGB')
        img_t = self.transform(img).unsqueeze(0).to(self.device)
        if self.encoder_fn is None:
            raise ValueError("No encoder_fn provided for on-the-fly feature extraction.")
        with torch.no_grad():
            feat = self.encoder_fn(img_t)
            if feat.ndim > 2:
                feat = torch.flatten(feat, start_dim=1)
            feat = feat.squeeze(0).cpu()
        return feat.float()

    def __getitem__(self, idx):
        # balanced random anchor sampling
        if random.random() < 0.5 and len(self.pos_list) > 0:
            anchor_list = self.pos_list; opp_list = self.neg_list
        else:
            anchor_list = self.neg_list; opp_list = self.pos_list
        anchor = random.choice(anchor_list)
        pos = random.choice(anchor_list)
        attempts = 0
        while pos == anchor and attempts < 10:
            pos = random.choice(anchor_list); attempts += 1
        neg = random.choice(opp_list)
        a_feat = self._load_patch_feature(anchor[0], anchor[1])
        p_feat = self._load_patch_feature(pos[0], pos[1])
        n_feat = self._load_patch_feature(neg[0], neg[1])
        return a_feat, p_feat, n_feat




