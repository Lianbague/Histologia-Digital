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

from ae_models import AutoEncoderCNN, AEConfigs

# Carrega de dades
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
        
        # Per a un AE basic, l'entrada (X) es igual a la sortida (Y) esperada (reconstruccio)
        return image, image 

def get_transforms():
    return transforms.Compose([
        transforms.Resize((256, 256)), 
        transforms.ToTensor(),         
        # Normalitzacio: Utilitzem valors estadistics (ImageNet) com a punt de partida.
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
    
    # Recorrer les seccions Cropped/PatID_Section
    for pat_id in negativa_patients:
        # Busca totes les carpetes amb el format PatID_X (p.ex., B22-03_0, B22-03_1)
        search_pattern = os.path.join(patches_root, f"{pat_id}_*")
        patient_folders = glob.glob(search_pattern)
        
        for folder in patient_folders:
            # Afegir tots els .png o .jpg dins de cada carpeta (assumim .png, pero pot ser .jpg)
            patch_files = glob.glob(os.path.join(folder, '*.png')) 
            # Si les imatges son jpg, canvieu '*.png' per '*.jpg'
            
            all_patch_paths.extend(patch_files)

    print(f"Total de patches NEGATIVA trobades per entrenament: {len(all_patch_paths)}")
    
    # Si el nombre es massa gran (mes de 100k), pot ser necessari limitar-lo per a la prova
    # if len(all_patch_paths) > 200000:
    #     all_patch_paths = all_patch_paths[:200000] 
    
    return all_patch_paths


if __name__ == "__main__":
    
    NEGATIVA_FILE = 'negativa_patients.txt' # Generat al Pas 1
    PATCHES_ROOT = '/export/fhome/maed/HelicoDataSet/CrossValidation/Cropped/'
    MODEL_SAVE_PATH = 'autoencoder_negativa_best.pth'
    CONFIG = '1' 
    
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-4 # Mes conservador per evitar divergencia
    NUM_EPOCHS = 30
    
    # Configuracio del dispositiu
    # Slurm s'encarrega d'assignar la GPU (DEVICE sera 'cuda:0' si hi ha GPU)
    DEVICE = torch.device("cuda:0")
    print(f"Utilitzant dispositiu: {DEVICE}")
    
    # Preparar les dades
    all_patch_paths = LoadCropped_Negativa(NEGATIVA_FILE, PATCHES_ROOT)
    
    transforms = get_transforms()
    dataset = PatchDataset(all_patch_paths, transform=transforms)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    
    # Configurar el model
    config = AEConfigs(config_id='1', input_channels=3)
    model = AutoEncoderCNN(
    net_paramsEnc=config.net_paramsEnc, 
    inputmodule_paramsDec=config.inputmodule_paramsDec, 
    net_paramsDec=config.net_paramsDec
    )
    model.to(DEVICE)
    
    # Train
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # Mean Squared Error (MSE) o L2 Loss es l'objectiu de reconstruccio
    criterion = nn.MSELoss() 
    
    print(f"Comencant l'entrenament (Epochs={NUM_EPOCHS}, Batches={len(dataloader)})...")
    
    best_loss = float('inf')
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        
        for i, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            
            optimizer.zero_grad()
            
            reconstructions = model(inputs)
            loss = criterion(reconstructions, targets)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(dataset)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {epoch_loss:.6f}")
        
        # Guardar el millor model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            # Guardem nomes els pesos
            torch.save(model.state_dict(), MODEL_SAVE_PATH) 
            print(f"Model guardat amb millor loss: {best_loss:.6f}")

    print("Entrenament finalitzat.")
