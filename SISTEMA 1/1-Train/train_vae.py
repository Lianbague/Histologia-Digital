# -*- coding: utf-8 -*-
import os
import glob
import sys
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

# Assegura't que tens aquest fitxer i classe disponibles
from _1_AE_train.ae_models import VariationalAutoEncoderCNN, AEConfigs

# ----------------------------------------------------------------------
# 1. DATASET OPTIMITZAT (Igual que l'AE - Màxima velocitat)
# ----------------------------------------------------------------------

class FastInMemoryDataset(Dataset):
    def __init__(self, file_paths):
        self.file_paths = file_paths
        print(f"Carregant {len(file_paths)} imatges a la RAM (Uint8)...")
        
        # MIDA FIXA 128x128
        self.data = torch.empty((len(file_paths), 3, 128, 128), dtype=torch.uint8)
        
        resize = transforms.Resize((128, 128))
        to_tensor = transforms.ToTensor() 
        
        for i, path in tqdm(enumerate(file_paths), total=len(file_paths)):
            try:
                with Image.open(path) as img:
                    img = img.convert('RGB')
                    img = resize(img)
                    # Guardem com a byte (0-255)
                    self.data[i] = (to_tensor(img) * 255).to(torch.uint8)
            except Exception as e:
                if i > 0: self.data[i] = self.data[i-1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Retornem només la imatge crua. El processament es fa al bucle.
        return self.data[idx]

def LoadCropped_Negativa(negativa_patients_file, patches_root):
    try:
        with open(negativa_patients_file, 'r') as f:
            negativa_patients = [line.strip() for line in f if line.strip()]
    except Exception as e:
        print(f"Error carregant llista de pacients: {e}")
        sys.exit(1)

    all_patch_paths = []
    for pat_id in negativa_patients:
        search_pattern = os.path.join(patches_root, f"{pat_id}_*")
        patient_folders = glob.glob(search_pattern)
        for folder in patient_folders:
            patch_files = glob.glob(os.path.join(folder, '*.png'))
            all_patch_paths.extend(patch_files)

    print(f"Total de patches NEGATIVA trobades: {len(all_patch_paths)}")
    return all_patch_paths

# ----------------------------------------------------------------------
# 2. FUNCIÓ DE PÈRDUA VAE
# ----------------------------------------------------------------------
def vae_loss_function(recon_x, x, mu, log_var, beta=1.0):
    """
    ReconLoss (MSE) + KLD Loss
    """
    ReconLoss = F.mse_loss(recon_x, x, reduction='sum')
    # KLD: -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return (ReconLoss + (beta * KLD)) / x.size(0)

# ----------------------------------------------------------------------
# 3. BLOC PRINCIPAL (MODIFICAT PER SEGURETAT)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    
    # --- CONFIGURACIÓ ---
    NEGATIVA_FILE = '/export/fhome/maed03/_0_data_preprocessing/negativa_patients.txt' 
    PATCHES_ROOT = '/export/fhome/maed/HelicoDataSet/CrossValidation/Cropped/'
    
    # IMPORTANT: Canviem el nom per no sobreescriure el model corrupte
    MODEL_SAVE_PATH = 'vae_negativa_128_fixed.pth' 
    
    CONFIG = '1'     
    
    BATCH_SIZE = 128  # Baixem una mica el batch per estabilitat (era 256)
    LEARNING_RATE = 1e-4 
    NUM_EPOCHS = 30
    LATENT_DIM = 128
    KLD_BETA = 1.0
    MAX_GRAD_NORM = 1.0 # Llindar per tallar els gradients (CLIPPING)
    
    DEVICE = torch.device("cuda:0")
    print(f"Entrenant VAE al dispositiu: {DEVICE}")
    
    # 1. OPTIMITZACIÓ CUDNN
    torch.backends.cudnn.benchmark = True
    
    # --- CÀRREGA DE DADES ---
    all_patch_paths = LoadCropped_Negativa(NEGATIVA_FILE, PATCHES_ROOT)
    dataset = FastInMemoryDataset(all_patch_paths)

    # --- DATALOADER ---
    dataloader = DataLoader(dataset, 
                            batch_size=BATCH_SIZE, 
                            shuffle=True, 
                            num_workers=2, 
                            persistent_workers=True, 
                            pin_memory=True)
    
    # --- MODEL VAE ---
    config = AEConfigs(config_id=CONFIG, input_channels=3)
    model = VariationalAutoEncoderCNN(
        inputmodule_paramsEnc=config.inputmodule_paramsEnc, 
        net_paramsEnc=config.net_paramsEnc, 
        inputmodule_paramsDec=config.inputmodule_paramsDec, 
        net_paramsDec=config.net_paramsDec,
        latent_dim=LATENT_DIM,
        img_size=128 # Assegurem que això coincideix amb el dataset
    )
    
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # --- MIXED PRECISION SETUP ---
    use_new_amp = hasattr(torch, 'amp') and hasattr(torch.amp, 'autocast')
    if use_new_amp:
        scaler = torch.amp.GradScaler('cuda')
    else:
        scaler = torch.cuda.amp.GradScaler()

    print(f"Comencant entrenament VAE (Safe Mode: Clipping Enabled)...")
    
    best_loss = float('inf')

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        
        for batch_uint8 in pbar:
            
            # 1. PREPARACIÓ DADES
            batch_uint8 = batch_uint8.to(DEVICE, non_blocking=True)
            inputs = batch_uint8.float().div(255.0)
            targets = inputs
            
            optimizer.zero_grad()
            
            # 2. FORWARD PASS AMB AMP
            if use_new_amp:
                with torch.amp.autocast('cuda'):
                    recon_batch, mu, log_var = model(inputs)
                    loss = vae_loss_function(recon_batch, targets, mu, log_var, beta=KLD_BETA)
            else:
                with torch.cuda.amp.autocast():
                    recon_batch, mu, log_var = model(inputs)
                    loss = vae_loss_function(recon_batch, targets, mu, log_var, beta=KLD_BETA)
            
            # --- SEGURETAT 1: Check NaNs a la Loss ---
            if torch.isnan(loss):
                print(f"\n[AVÍS] Loss NaN detectada al batch! Saltant actualització.")
                continue # Saltem aquest batch, no actualitzem pesos
            
            # 3. BACKPROPAGATION ESCALAT
            scaler.scale(loss).backward()
            
            # --- SEGURETAT 2: GRADIENT CLIPPING AMB SCALER ---
            # Primer hem de des-escalar els gradients per poder-los tallar correctament
            scaler.unscale_(optimizer)
            
            # Tallem els gradients que superin 1.0
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=MAX_GRAD_NORM)
            
            # Ara fem el step (si els gradients no eren inf/nan, l'unscale ja ho ha comprovat)
            scaler.step(optimizer)
            scaler.update()
            
            # Log
            loss_val = loss.item()
            running_loss += loss_val * inputs.size(0)
            pbar.set_postfix({'vae_loss': loss_val})

        epoch_loss = running_loss / len(dataset)
        print(f"Fi Epoch {epoch+1}. VAE Loss Mitjana: {epoch_loss:.6f}")
        
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"Model VAE guardat a {MODEL_SAVE_PATH} (Loss: {best_loss:.6f})")

    print("Entrenament VAE segur finalitzat.")