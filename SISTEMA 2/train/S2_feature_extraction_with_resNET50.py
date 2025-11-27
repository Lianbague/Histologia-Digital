
from S2_MIL.S2_feature_extraction_with_AE import ImageBagDataset

import pandas as pd
import os
import torch
from typing import List, Tuple
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import torchvision.models as models
import torch.nn as nn

import glob
from PIL import Image
import numpy as np
from tqdm import tqdm


IMAGE_DIR = '/export/fhome/maed/HelicoDataSet/CrossValidation/Cropped'
diagnosis_file = '/export/fhome/maed/HelicoDataSet/PatientDiagnosis.csv'

OUTPUT_DIR = '/export/fhome/maed03/Features_ResNet' 
os.makedirs(OUTPUT_DIR, exist_ok=True)

transform = transforms.Compose([
        transforms.Resize((224, 224)), # ResNet standard input size
        transforms.ToTensor(),              
        transforms.Normalize(               
            mean=[0.485, 0.456, 0.406],     
            std=[0.229, 0.224, 0.225]       
        )
    ])

# Identificar pacients
patient_ids = set([os.path.basename(f).split('_')[0] for f in glob.glob(os.path.join(IMAGE_DIR, '*'))]) # El que fa aixo es buscar totes les carpetes dins de RAW_DATA_ROOT i agafar la primera part del nom de la carpeta abans del guio baix (_)
# patient_ids is a set of unique patient identifiers extracted from folder names in IMAGE_DIR with the format 'patientID_something'.

# Load the pre-trained ResNet-50 model
pretrained_ResNET50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

# Freeze all the parameters in the model
for param in pretrained_ResNET50.parameters():
    param.requires_grad = False

# Set the model to evaluation mode (important for Batch Norm and Dropout)
pretrained_ResNET50.eval()
# Remove the final fully connected layer
feature_extractor = nn.Sequential(*list(pretrained_ResNET50.children())[:-1]) 

# Put the model on the correct device (GPU is preferred)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
feature_extractor.to(device)

print(f"Extraient caracteristiques de {len(patient_ids)} pacients...")

with torch.no_grad():
        for pat_id in tqdm(patient_ids):
            # Buscar tots els patches del pacient
            pat_sections = glob.glob(os.path.join(IMAGE_DIR, f"{pat_id}_*"))
            all_patches = []
            for sec in pat_sections:
                all_patches.extend(glob.glob(os.path.join(sec, '*.png')))
            
            if not all_patches: continue

            # Crear DataLoader per les imatges del pacient 
            dataset = ImageBagDataset(all_patches, transform=transform)
            dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

            patient_features = []
            for imgs in dataloader:
                imgs = imgs.to(device)
                # Passar pel encoder
                features = feature_extractor(imgs) 
                # Aplanar a un vector: [Batch, Features]
                features = torch.flatten(features, start_dim=1)
                patient_features.append(features.cpu())
            
            # Concatenar totes les caracteristiques del pacient
            patient_tensor = torch.cat(patient_features, dim=0)  
            
            # Save
            torch.save(patient_tensor, os.path.join(OUTPUT_DIR, f"{pat_id}.pt"))







