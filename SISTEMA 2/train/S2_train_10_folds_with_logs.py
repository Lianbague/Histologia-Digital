# S2_train_mil_kfold.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, f1_score
import json

from S2_MIL.S2_models import NeuralNetwork_withAttention


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
CSV_PATH = '/export/fhome/maed/HelicoDataSet/PatientDiagnosis.csv'
NUM_FOLDS = 10
NUM_EPOCHS = 30
BATCH_SIZE = 1 # En MIL standard sol ser 1 (bags de mida variable)


FOLDS_JSON_PATH = '/fhome/maed03/_2_fix_threshold/folds_distribution_10folds.json'

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

def train_system2_10folds(feature_extractor_model):
    print("Inicilitzant entrenament Sistema 2...")
    
    if feature_extractor_model == 'ResNet':
        FEATURES_DIR = '/export/fhome/maed03/Features_ResNet'
        print("Utilitzant caracteristiques de ResNet")
    
    elif feature_extractor_model == 'AE':
        FEATURES_DIR = '/export/fhome/maed03/Features_AE'
        print("Utilitzant caracteristiques d'AutoEncoder")
    
    elif feature_extractor_model == 'Triplet':
        FEATURES_DIR = '/export/fhome/maed03/Features_Triplet128'
        print("Utilitzant caracteristiques de Contrastive Learning (Triplet)")
    
    else:
        raise ValueError("Model no reconegut. Usa 'ResNet', 'AE' o 'Triplet'")

    df = pd.read_csv(CSV_PATH)

    available_pats = [f.replace('.pt', '') for f in os.listdir(FEATURES_DIR) if f.endswith('.pt')]
    # Filtrar per assegurar que tenim etiqueta per a cada pacient
    valid_patients = [p for p in available_pats if p in df['CODI'].values]
    
    print(f"Pacients valids trobats: {len(valid_patients)}")


    # 1. CARGAR DISTRIBUCION DE FOLDS
    try:
        with open(FOLDS_JSON_PATH, 'r') as f:
            folds_data = json.load(f)
        
        if len(folds_data) != NUM_FOLDS:
            print(f"ADVERTENCIA: El JSON contiene {len(folds_data)} folds, pero NUM_FOLDS está establecido en {NUM_FOLDS}.")

    except FileNotFoundError:
        print(f"ERROR: Archivo de folds no encontrado en {FOLDS_JSON_PATH}")
        return
    except json.JSONDecodeError:
        print(f"ERROR: El archivo {FOLDS_JSON_PATH} no es un JSON válido.")
        return

    
    # DEFINIR HIPERPARAMETRES DEL MODEL
    # Verificar dimensio dels .pt carregant un de prova
    temp_feat = torch.load(os.path.join(FEATURES_DIR, valid_patients[0] + ".pt"), weights_only=True)
    INPUT_DIM = temp_feat.shape[1]
    print(f"Dimensio de entrada detectada: {INPUT_DIM}")
    
    ATTENTION_BRANCHES = 1
    attention_params = {
        'in_features': INPUT_DIM,
        'decom_space': 128, # Espai latent de atencion (L)
        'ATTENTION_BRANCHES': ATTENTION_BRANCHES
    }
    
    classifier_params = {
        'in_features': INPUT_DIM * ATTENTION_BRANCHES,
        'out_features': 1 # Binari
    }

    # Bucle K-Fold
    #kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
    fold_results = []

    loss_log_for_folds = dict() 
    loss_log = {"train": []}
    roc_log_for_folds = dict() 

    for fold, fold_data in enumerate(folds_data):
        print(f"\n--- FOLD {fold+1}/{NUM_FOLDS} ---")

        # Get patient IDs from JSON
        train_pats_json = fold_data['train']
        val_pats_json = fold_data['test']

        # Only keep patients from the JSON split if they are present in the valid_patients list
        train_pats = [p for p in train_pats_json if p in valid_patients]
        val_pats = [p for p in val_pats_json if p in valid_patients]
        
        train_ds = PatientFeatureDataset(train_pats, FEATURES_DIR, df)
        val_ds = PatientFeatureDataset(val_pats, FEATURES_DIR, df)
        
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
        
        # Instanciar Model Nou per cada Fold
        model = NeuralNetwork_withAttention(
            input_dim=INPUT_DIM,       # 32768
            project_dim=512,           # Reduim a 512
            decom_space=128,
            attention_branches=1,
            attention_type='GatedAttention'
        ).to(DEVICE)
        

        optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
        criterion = nn.BCELoss()
        
        best_auc = 0.0
        loss_log = {"train": []} 

        best_val_labels = None 
        best_val_probs = None 
        for epoch in range(NUM_EPOCHS):
            model.train()
            train_loss = 0 

            
            for features, label in train_loader:
                features, label = features.to(DEVICE), label.to(DEVICE)
                
                optimizer.zero_grad()
                probs, _ = model(features) # Retorna output, attention_weights
                loss = criterion(probs, label)
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
                    probs, _ = model(features)
                    val_probs.append(probs.item())
                    val_labels.append(label.item())
            
            try:
                auc = roc_auc_score(val_labels, val_probs)
            except:
                auc = 0.5 # Si falla per etiquetes constants
                
            if auc > best_auc:
                best_auc = auc

                best_val_labels = val_labels.copy() 
                best_val_probs = val_probs.copy() 

                # Guardar millor model del fold
                save_name = f"S2_best_model_{feature_extractor_model}_fold{fold+1}.pth"
                torch.save(model.state_dict(), save_name)
            
            if (epoch+1) % 5 == 0:
                print(f"  Epoch {epoch+1}: Loss {train_loss/len(train_loader):.4f} | Val AUC {auc:.4f}")

            loss_log["train"].append(train_loss/len(train_loader)) 
        loss_log_for_folds[f"fold_{fold+1}"] = loss_log 
        roc_log_for_folds[f"fold_{fold+1}"] = {
            "labels": best_val_labels,
            "probs": best_val_probs
             } 
        
        print(f"  >> Millor AUC Fold {fold+1}: {best_auc:.4f}")
        fold_results.append(best_auc)


    print(f"\nResultats Finals Sistema 2: {np.mean(fold_results):.4f} +/- {np.std(fold_results):.4f}")
    return loss_log_for_folds, roc_log_for_folds  
    



    