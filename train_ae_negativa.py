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

# ==============================================================================
# A. DEFINICIONS DEL MODEL (EXTRETES DE AEExample_Script..py)
# ==============================================================================

# -----------------
# CLASSES AUXILIARS DEL MODEL
# -----------------
class ConvBlock(nn.Module):
    # Bloc de Convolucio per a l'Encoder
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                              stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class TConvBlock(nn.Module):
    # Bloc de Transposada de Convolucio per al Decoder
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding=0):
        super(TConvBlock, self).__init__()
        self.tconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, 
                                        stride=stride, padding=padding, output_padding=output_padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.tconv(x)))

# -----------------
# CLASSE PRINCIPAL: AutoEncoderCNN
# -----------------
class AutoEncoderCNN(nn.Module):
    def __init__(self, inputmodule_paramsDec, net_paramsEnc, net_paramsDec):
        super(AutoEncoderCNN, self).__init__()

        # ENCODER
        layers = []
        in_channels = 3 # RGB input
        for block in net_paramsEnc['block_configs']:
            layers.append(ConvBlock(in_channels, block[0], block[1], block[2], block[3]))
            in_channels = block[0]
        self.encoder = nn.Sequential(*layers)

        # DECODER
        layers = []
        in_channels = inputmodule_paramsDec['num_input_channels'] 
        for i, block in enumerate(net_paramsDec['block_configs']):
            is_last = (i == len(net_paramsDec['block_configs']) - 1)
            
            # ConvTranspose2d (amb possible output_padding per igualar les dimensions)
            output_p = block[4] if len(block) > 4 else 0
            tconv = nn.ConvTranspose2d(in_channels, block[0], kernel_size=block[1], 
                                       stride=block[2], padding=block[3], 
                                       output_padding=output_p) 
                        
            if is_last:
                # Per a la darrera capa, l'output_padding ha de ser 1 (per 128->256) o el necessari per arribar a 256.
                # Utilitzem 'output_padding=1' si el bloc no el defineix, es un truc comu per restaurar la mida.
                final_output_padding = 1 if output_p == 0 else output_p
                
                tconv_final = nn.ConvTranspose2d(in_channels, block[0], kernel_size=block[1], 
                                                 stride=block[2], padding=block[3], 
                                                 output_padding=final_output_padding)
                
                layers.append(nn.Sequential(tconv_final, nn.Sigmoid()))
                
            elif not is_last:
                layers.append(TConvBlock(in_channels, block[0], block[1], block[2], block[3], output_padding=output_p))
                in_channels = block[0]
        
        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        encoded = self.encoder(x)
        reconstructed = self.decoder(encoded)
        return reconstructed

# -----------------
# FUNCIo DE CONFIGURACIo (AEConfigs)
# -----------------
def AEConfigs(Config):
    net_paramsEnc, net_paramsDec, inputmodule_paramsDec = {}, {}, {}

    if Config=='1':
        # ENCODER: 256 -> 128 -> 64 -> 32 -> 16 -> 8
        net_paramsEnc['block_configs'] = [
            [32, 3, 2, 1],  
            [64, 3, 2, 1], 
            [128, 3, 2, 1], 
            [256, 3, 2, 1], 
            [512, 3, 2, 1]  
        ]
        
        # DECODER: 8 -> 16 -> 32 -> 64 -> 128 -> 256 (El 5e element es l'output_padding)
        net_paramsDec['block_configs'] = [
            [256, 3, 2, 1, 1], 
            [128, 3, 2, 1, 1], 
            [64, 3, 2, 1, 1],  
            [32, 3, 2, 1, 1],  
            [3, 3, 2, 1, 1]    
        ]
        
        inputmodule_paramsDec['num_input_channels'] = net_paramsEnc['block_configs'][-1][0] # 512

    return net_paramsEnc, net_paramsDec, inputmodule_paramsDec

# ==============================================================================
# B. CaRREGA DE DADES (ADAPTAT)
# ==============================================================================

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
    # Es crucial que la normalitzacio sigui 'sensata'
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
    
    # Recorrer les seccions Cropped/PatID_Section#
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


# ==============================================================================
# C. BUCLE D'ENTRENAMENT
# ==============================================================================

if __name__ == "__main__":
    
    # --- 0. PARaMETRES ---
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
    
    # --- 1. CaRREGA DE DADES ---
    all_patch_paths = LoadCropped_Negativa(NEGATIVA_FILE, PATCHES_ROOT)
    
    transforms = get_transforms()
    dataset = PatchDataset(all_patch_paths, transform=transforms)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    
    # --- 2. CONFIGURACIo DEL MODEL ---
    net_paramsEnc, net_paramsDec, inputmodule_paramsDec = AEConfigs(CONFIG)
    model = AutoEncoderCNN(inputmodule_paramsDec, net_paramsEnc, net_paramsDec).to(DEVICE)
    
    # --- 3. ENTRENAMENT ---
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
