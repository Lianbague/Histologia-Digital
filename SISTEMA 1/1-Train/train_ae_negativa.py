# -*- coding: utf-8 -*-
import os
import glob
import sys
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from AE_train_1.ae_models import AutoEncoderCNN, AEConfigs

# ----------------------------------------------------------------------
# 1. DATASET OPTIMITZAT (Retorna uint8 per estalviar ample de banda)
# ----------------------------------------------------------------------

class FastInMemoryDataset(Dataset):
    def __init__(self, file_paths):
        self.file_paths = file_paths
        print(f"Carregant {len(file_paths)} imatges a la RAM (Uint8)...")
        
        # Reservem memòria: [N, 3, 256, 256] uint8
        # Si tens problemes de memòria, canvia 256 per 128
        self.data = torch.empty((len(file_paths), 3, 128, 128), dtype=torch.uint8)
        
        resize = transforms.Resize((128, 128))
        to_tensor = transforms.ToTensor() 
        
        # Càrrega inicial amb barra de progrés
        for i, path in tqdm(enumerate(file_paths), total=len(file_paths)):
            try:
                with Image.open(path) as img:
                    img = img.convert('RGB')
                    img = resize(img)
                    # ToTensor -> float [0,1]. Multipliquem per 255 -> uint8 [0,255]
                    self.data[i] = (to_tensor(img) * 255).to(torch.uint8)
            except Exception as e:
                if i > 0: self.data[i] = self.data[i-1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # RETORNEM NOMÉS LA IMATGE EN BYTES (uint8).
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
# 2. BLOC PRINCIPAL
# ----------------------------------------------------------------------
if __name__ == "__main__":
    
    # --- CONFIGURACIÓ ---
    NEGATIVA_FILE = 'negativa_patients.txt' 
    PATCHES_ROOT = '/export/fhome/maed/HelicoDataSet/CrossValidation/Cropped/'
    MODEL_SAVE_PATH = 'ae_L1Loss_128.pth'
    CONFIG = '1'    
    
    BATCH_SIZE = 256 
    LEARNING_RATE = 1e-4 
    NUM_EPOCHS = 30
    
    DEVICE = torch.device("cuda:0")
    print(f"Utilitzant dispositiu: {DEVICE}")
    
    # 1. OPTIMITZACIÓ CUDNN (Això SÍ que funciona en Pascal)
    torch.backends.cudnn.benchmark = True
    
    # --- CÀRREGA ---
    all_patch_paths = LoadCropped_Negativa(NEGATIVA_FILE, PATCHES_ROOT)
    dataset = FastInMemoryDataset(all_patch_paths)

    # --- DATALOADER ---
    dataloader = DataLoader(dataset, 
                            batch_size=BATCH_SIZE, 
                            shuffle=True, 
                            num_workers=2, 
                            persistent_workers=True, 
                            pin_memory=True) 
    
    # --- MODEL ---
    config = AEConfigs(config_id='1', input_channels=3)
    model = AutoEncoderCNN(
        net_paramsEnc=config.net_paramsEnc, 
        inputmodule_paramsDec=config.inputmodule_paramsDec, 
        net_paramsDec=config.net_paramsDec
    )
    
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.L1Loss()    
    
    # --- NOTA: HE ELIMINAT torch.compile PERQUÈ LA GPU NO HO SUPORTA ---
    
    # --- CONFIGURACIÓ MIXED PRECISION ---
    # Comprovar versió de PyTorch
    use_new_amp = hasattr(torch, 'amp') and hasattr(torch.amp, 'autocast')
    
    if use_new_amp:
        scaler = torch.amp.GradScaler('cuda')
    else:
        scaler = torch.cuda.amp.GradScaler()

    print(f"Comencant l'entrenament...")
    
    best_loss = float('inf')

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        
        for batch_uint8 in pbar:
            
            # 1. Movem bytes a GPU
            batch_uint8 = batch_uint8.to(DEVICE, non_blocking=True)
            
            # 2. Convertim a float i normalitzem A LA GPU
            inputs = batch_uint8.float().div(255.0)
            
            # 3. Target == Input
            targets = inputs
            
            optimizer.zero_grad()
            
            # 4. Mixed Precision
            if use_new_amp:
                with torch.amp.autocast('cuda'):
                    reconstructions = model(inputs)
                    loss = criterion(reconstructions, targets)
            else:
                with torch.cuda.amp.autocast():
                    reconstructions = model(inputs)
                    loss = criterion(reconstructions, targets)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            loss_val = loss.item()
            running_loss += loss_val * inputs.size(0)
            pbar.set_postfix({'loss': loss_val})

        epoch_loss = running_loss / len(dataset)
        print(f"Fi Epoch {epoch+1}. Loss Mitjana: {epoch_loss:.6f}")
        
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"Model guardat (Loss: {best_loss:.6f})")

    print("Entrenament finalitzat.")