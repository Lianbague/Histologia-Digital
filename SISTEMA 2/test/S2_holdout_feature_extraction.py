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
from _1_AE_train.ae_models import AutoEncoderCNN, AEConfigs


# ELIGE EL MODO AQUÍ: 'ResNet', 'Triplet' o 'AE'
MODEL_TYPE = 'AE' 

# RUTAS DE LOS MODELOS
AE_PATH = "/export/fhome/maed03/_1_AE_train/ae_L1Loss_128.pth"
TRIPLET_PATH = "/export/fhome/maed03/S2_triplet_model_final.pth"

# DATOS DE ENTRADA
IMAGE_DIR = '/export/fhome/maed/HelicoDataSet/HoldOut'
diagnosis_file = '/export/fhome/maed/HelicoDataSet/PatientDiagnosis.csv'

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Clase para Triplet
class ResNetEmbedding(nn.Module):
    def __init__(self, embedding_dim=128):
        super(ResNetEmbedding, self).__init__()
        
        # Carreguem ResNet50 preentrenada
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        # Ens quedem amb tot menys la capa final
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # Congelar ResNet (nomes entrenem el projector)
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # Capa de projeccio
        self.projector = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, embedding_dim)
        )

    def forward(self, x):
        # Mode eval per al backbone
        self.backbone.eval()
        with torch.no_grad():
            features = self.backbone(x)
        
        embeddings = self.projector(features)
        
        # Normalitzacio L2: important per estabilitzar distancia euclidiana
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings

transform = transforms.Compose([
        transforms.Resize((224, 224)), # ResNet standard input size
        transforms.ToTensor(),              
        transforms.Normalize(               
            mean=[0.485, 0.456, 0.406],     
            std=[0.229, 0.224, 0.225]       
        )
    ])

class ImageBagDataset(Dataset):
    def __init__(self, patch_paths, transform=None):
        self.patch_paths = patch_paths
        self.transform = transform
    def __len__(self): return len(self.patch_paths)
    def __getitem__(self, idx):
        try:
            img = Image.open(self.patch_paths[idx]).convert('RGB')
            if self.transform: img = self.transform(img)
            return img
        except:
            return None
            
def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if not batch: return None
    return torch.utils.data.dataloader.default_collate(batch)


def extract_features():
    print(f"--- Iniciant Extracció de Caracteristiques: Modo {MODEL_TYPE} ---")
    
    # 1. Configurar Modelo y Directorio de Salida
    if MODEL_TYPE == 'ResNet':
        OUTPUT_DIR = '/export/fhome/maed03/Features_HoldOut_ResNet'
        
        # ResNet50 estándar sin la capa final
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        resnet.fc = nn.Identity() 
        model = resnet
        img_size = 224
        
    elif MODEL_TYPE == 'Triplet':
        OUTPUT_DIR = '/export/fhome/maed03/Features_HoldOut_Triplet128'
        
        # Modelo Triplet entrenado
        model = ResNetEmbedding(embedding_dim=128)
        if os.path.exists(TRIPLET_PATH):
            model.load_state_dict(torch.load(TRIPLET_PATH, map_location=DEVICE))
        
        else:
            print(f"ERROR: No se encuentra {TRIPLET_PATH}")
            sys.exit(1)
            
        img_size = 224

    elif MODEL_TYPE == 'AE':
        OUTPUT_DIR = '/export/fhome/maed03/Features_HoldOut_AE'
        
        # Cargar Autoencoder
        config = AEConfigs(config_id='1', input_channels=3)
        
        full_model = AutoEncoderCNN(
            net_paramsEnc=config.net_paramsEnc,
            inputmodule_paramsDec=config.inputmodule_paramsDec,
            net_paramsDec=config.net_paramsDec
        )
        
        if os.path.exists(AE_PATH):
            full_model.load_state_dict(torch.load(AE_PATH, map_location=DEVICE))
            print("Pesos AE cargados correctamente.")
        else:
            print(f"ERROR: No se encuentra {AE_PATH}")
            sys.exit(1)
            
        # Nos quedamos solo con el encoder
        model = full_model.encoder
        img_size = 256 # El AE solía usar 256

    else:
        raise ValueError("MODEL_TYPE debe ser 'ResNet', 'Triplet' o 'AE'")

    # Preparar modelo
    model.to(DEVICE)
    model.eval()
    
    # Crear directorio
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Directorio de salida: {OUTPUT_DIR}")

    # 2. Transformaciones
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)), 
        transforms.ToTensor(),              
        transforms.Normalize(                
            mean=[0.485, 0.456, 0.406],      
            std=[0.229, 0.224, 0.225]        
        )
    ])

    # 3. Bucle de Extracción
    # IMPORTANTE: Usamos sorted() para garantizar consistencia con visualización
    patient_ids = sorted(list(set([os.path.basename(f).split('_')[0] 
                                   for f in glob.glob(os.path.join(IMAGE_DIR, '*'))])))

    print(f"Procesant {len(patient_ids)} pacients...")

    with torch.no_grad():
        for pat_id in tqdm(patient_ids):
            save_path = os.path.join(OUTPUT_DIR, f"{pat_id}.pt")
            
            # Buscar carpetas del paciente (ORDENADAS)
            pat_sections = sorted(glob.glob(os.path.join(IMAGE_DIR, f"{pat_id}_*")))
            all_patches = []
            for sec in pat_sections:
                # Buscar parches (ORDENADOS)
                patches = sorted(glob.glob(os.path.join(sec, '*.png')))
                all_patches.extend(patches)
            
            if not all_patches: continue

            # DataLoader
            dataset = ImageBagDataset(all_patches, transform=transform)
            dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4, collate_fn=collate_fn)

            patient_features = []
            for imgs in dataloader:
                if imgs is None: continue
                imgs = imgs.to(DEVICE)
                
                # Extracción
                features = model(imgs) 
                
                # Si es ResNet/AE, aplanamos [B, C, H, W] -> [B, Features]
                # Si es Triplet, ya sale aplanado [B, 128]
                if len(features.shape) > 2:
                    features = torch.flatten(features, start_dim=1)
                
                patient_features.append(features.cpu())
            
            if patient_features:
                patient_tensor = torch.cat(patient_features, dim=0)
                torch.save(patient_tensor, save_path)

    print("Extraccio completada amb exit!")


if __name__ == '__main__':
    extract_features()




