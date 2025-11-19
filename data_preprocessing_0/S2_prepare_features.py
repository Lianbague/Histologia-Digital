"""Converteix imatges .png a tensors .pt (per no pasar les imatges cada cop ja que seria molt lent)"""

import torch
import os
import glob
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
from tqdm import tqdm
from AE_train_1.ae_models import AutoEncoderCNN, AEConfigs

class ImageBagDataset(Dataset):
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
        
        return image

def extract_features():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Carrega el model AE preentrenat (Sistema 1)
    MODEL_PATH = 'autoencoder_negativa_best_L1Loss.pth'
    config = AEConfigs(config_id='1', input_channels=3)

    # Inicialitzem el model sencer pero només utilitzarem l'encoder per extreure característiques
    full_model = AutoEncoderCNN(
        inputmodule_paramsEnc=config.inputmodule_paramsEnc,
        net_paramsEnc=config.net_paramsEnc,
        inputmodule_paramsDec=config.inputmodule_paramsDec, 
        net_paramsDec=config.net_paramsDec 
    )

    full_model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    encoder = full_model.encoder.to(DEVICE) # Nomes l'encoder
    encoder.eval()

    RAW_DATA_ROOT = '/export/fhome/maed/HelicoDataSet/CrossValidation/Cropped'
    OUTPUT_DIR = '/export/fhome/maed03/Features_AE' 
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Transformacions (resize/tensor/normalitzacio)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Identificar pacients
    patient_ids = set([os.path.basename(f).split('_')[0] for f in glob.glob(os.path.join(RAW_DATA_ROOT, '*'))])
    
    print(f"Extraient caracteristiques de {len(patient_ids)} pacients...")

    with torch.no_grad():
        for pat_id in tqdm(patient_ids):
            # Buscar tots els patches del pacient
            pat_sections = glob.glob(os.path.join(RAW_DATA_ROOT, f"{pat_id}_*"))
            all_patches = []
            for sec in pat_sections:
                all_patches.extend(glob.glob(os.path.join(sec, '*.png')))
            
            if not all_patches: continue

            # Crear DataLoader per les imatges del pacient 
            dataset = ImageBagDataset(all_patches, transform=transform)
            loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

            patient_features = []
            for imgs in loader:
                imgs = imgs.to(DEVICE)
                # Passar pel encoder
                features = encoder(imgs) 
                # El encoder retorna [Batch, Channels, H, W]
                # Aplanar a un vector: [Batch, Features]
                features = torch.flatten(features, start_dim=1)
                patient_features.append(features.cpu())
            
            # Concatenar totes les caracteristiques del pacient
            # Resultat: Tensor de mida [Num_Patches, Dimension_Latente]
            patient_tensor = torch.cat(patient_features, dim=0)
            
            # Save
            torch.save(patient_tensor, os.path.join(OUTPUT_DIR, f"{pat_id}.pt"))

if __name__ == '__main__':
    extract_features()