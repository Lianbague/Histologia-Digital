# train_mil_kfold.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, f1_score

from AE_train_1.ae_models import AttentionMIL 

NUM_EPOCHS = 20

class PatientFeatureDataset(Dataset):
    def __init__(self, patient_ids, features_dir, df_diagnosis):
        self.patient_ids = patient_ids
        self.features_dir = features_dir
        self.df = df_diagnosis
    
    def __len__(self): return len(self.patient_ids)
    
    def __getitem__(self, idx):
        pat_id = self.patient_ids[idx]
        # Carregar el tensor calculat previament de caracteristiques
        features_path = os.path.join(self.features_dir, f"{pat_id}.pt")
        features = torch.load(features_path, weights_only=True)
        
        # Obtenir etiqueta (Bacteria = 1, Negativa = 0)
        label_str = self.df[self.df['CODI'] == pat_id]['DENSITAT'].values[0]
        label = 1.0 if label_str != 'NEGATIVA' else 0.0
        
        return features, torch.tensor([label], dtype=torch.float32)

def train_system2():
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    FEATURES_DIR = '/export/fhome/maed03/Features_AE' 
    CSV_PATH = '/export/fhome/maed/HelicoDataSet/PatientDiagnosis.csv'
    
    df = pd.read_csv(CSV_PATH)
    all_patients = [f.replace('.pt', '') for f in os.listdir(FEATURES_DIR)]
    # Filtrar per assegurar que tenim etiqueta per a cada pacient
    all_patients = [p for p in all_patients if p in df['CODI'].values]
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    results = []

    print("Inicilitzant entrenament Sistema 2 (MIL+Attention)...")

    for fold, (train_idx, val_idx) in enumerate(kf.split(all_patients)):
        print(f"\n--- FOLD {fold+1} ---")
        
        train_pats = [all_patients[i] for i in train_idx]
        val_pats = [all_patients[i] for i in val_idx]
        
        train_ds = PatientFeatureDataset(train_pats, FEATURES_DIR, df)
        val_ds = PatientFeatureDataset(val_pats, FEATURES_DIR, df)
        
        # Batch size = 1 perque cada pacient te diferent numero de patches
        train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
        
        # Detectar dimensio d'entrada
        sample_feat, _ = train_ds[0]
        input_dim = sample_feat.shape[1] # Dimensionalitat de les caracteristiques extretes
        
        model = AttentionMIL(input_dim=input_dim).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
        criterion = nn.BCELoss() # Binary Cross Entropy
        
        for epoch in range(NUM_EPOCHS): 
            model.train()
            train_loss = 0
            for features, label in train_loader:
                features, label = features.to(DEVICE), label.to(DEVICE)
                
                optimizer.zero_grad()
                probs, _, _ = model(features)
                loss = criterion(probs, label)
                
                # Gradient Accumulation 
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # Validacio
            model.eval()
            val_probs = []
            val_labels = []
            with torch.no_grad():
                for features, label in val_loader:
                    features = features.to(DEVICE)
                    probs, _, _ = model(features)
                    val_probs.append(probs.item())
                    val_labels.append(label.item())
            
            auc = roc_auc_score(val_labels, val_probs)
            print(f"Epoch {epoch+1}: Loss={train_loss/len(train_loader):.4f}, Val AUC={auc:.4f}")
        
        results.append(auc)
        # Guardar model del fold
        torch.save(model.state_dict(), f"mil_model_fold{fold+1}.pth")

    print(f"\nResultats finals 5-Fold AUC: {np.mean(results):.4f} +/- {np.std(results):.4f}")

if __name__ == '__main__':
    train_system2()