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
import torch.optim as optim

from AE_train_1.ae_models import VariationalAutoEncoderCNN, AEConfigs

class PatchDataset(Dataset):
    def __init__(self, file_paths, transform=None):
        self.file_paths = file_paths
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        image = Image.open(img_path).convert('RGB') 
        
        if self.transform:
            image = self.transform(image)
        
        # Per a un AE/VAE, l'entrada (X) es igual a la sortida (Y) esperada (reconstruccio)
        return image, image 

def get_transforms():
    """ Transformacions d'entrenament, incloent normalització. """
    return transforms.Compose([
        transforms.Resize((256, 256)), 
        transforms.ToTensor(),       
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ])

def LoadCropped_Negativa(negativa_patients_file, patches_root):
    """ Carga la llista de paths de totes les patches NEGATIVA. """
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

    print(f"Total de patches NEGATIVA trobades per entrenament: {len(all_patch_paths)}")
    return all_patch_paths


def vae_loss_function(recon_x, x, mu, log_var, beta=1.0):
    """
    Calcula la perdua VAE (combinada): Perdua de Reconstruccio + Perdua KLD.
    beta: Factor per ponderar la KLD (si beta=1, es VAE estandard)
    """
    #Perdua de Reconstruccio (MSE/L2, sumada sobre el batch)
    # Usem MSE per ser consistents amb l'avaluacio d'anomalies
    ReconLoss = F.mse_loss(recon_x, x, reduction='sum')

    # Perdua de Divergencia KL
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    # Perdua total
    return ReconLoss + (beta * KLD)


if __name__ == "__main__":
    
    # Configuració de paths
    NEGATIVA_FILE = '/export/fhome/maed03/data_preprocessing_0/negativa_patients.txt' 
    PATCHES_ROOT = '/export/fhome/maed/HelicoDataSet/CrossValidation/Cropped/'
    MODEL_SAVE_PATH = '/export/fhome/maed03/vae_negativa_best.pth' 
    CONFIG = '1' 
    
    # Hiperparametres
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-4 
    NUM_EPOCHS = 30 
    LATENT_DIM = 128 # Mida de l'espai latent del VAE (provar 64, 128, 256)
    KLD_BETA = 1.0   # Ponderador de la perdua KLD
    
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Entrenant VAE. Utilitzant dispositiu: {DEVICE}")
    
    # Preparar les dades
    all_patch_paths = LoadCropped_Negativa(NEGATIVA_FILE, PATCHES_ROOT)
    transforms = get_transforms()
    dataset = PatchDataset(all_patch_paths, transform=transforms)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    
    # Configurar el model
    config = AEConfigs(config_id=CONFIG, input_channels=3)
    model = VariationalAutoEncoderCNN(
        inputmodule_paramsEnc=config.inputmodule_paramsEnc, 
        net_paramsEnc=config.net_paramsEnc, 
        inputmodule_paramsDec=config.inputmodule_paramsDec, 
        net_paramsDec=config.net_paramsDec,
        latent_dim=LATENT_DIM
    )
    model.to(DEVICE) 
    
    # Optimitzador
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print(f"Comencant l'entrenament del VAE (Epochs={NUM_EPOCHS}, Batches={len(dataloader)})...")
    
    best_loss = float('inf')
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        
        for i, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            
            optimizer.zero_grad()
            
            # El model VAE retorna 3 valors 
            reconstructions, mu, log_var = model(inputs)
            
            # Usem la nova funció de perdua
            loss = vae_loss_function(reconstructions, targets, mu, log_var, beta=KLD_BETA)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)

        # Calculem la perdua mitjana per imatge
        epoch_loss = running_loss / len(dataset)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, VAE Loss: {epoch_loss:.6f}")
        
        # Guardar el millor model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH) 
            print(f"Model VAE guardat amb millor loss: {best_loss:.6f}")

    print("Entrenament VAE finalitzat.")