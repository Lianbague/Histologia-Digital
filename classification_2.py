# -*- coding: utf-8 -*-
"""
Sistema 2: ResNet + Triplet Loss
Basado en el pipeline del Sistema 1 (AE)
"""
import os
import glob
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc, f1_score, accuracy_score, confusion_matrix

# =========================
# 1️⃣ Dataset de Tripletas
# =========================
class TripletPatchDataset(Dataset):
    """
    Dataset para entrenamiento con Triplet Loss:
    Anchor, Positive (misma clase), Negative (otra clase)
    """
    def __init__(self, positive_paths, negative_paths, transform=None):
        self.positive_paths = positive_paths
        self.negative_paths = negative_paths
        self.transform = transform

        # Crear lista de anchors con sus positivos correspondientes
        self.anchor_positive_pairs = []
        for p in positive_paths:
            # Para simplificar, cada anchor tiene un positivo aleatorio distinto
            pos = np.random.choice([x for x in positive_paths if x != p])
            self.anchor_positive_pairs.append((p, pos))

    def __len__(self):
        return len(self.anchor_positive_pairs)

    def __getitem__(self, idx):
        anchor_path, positive_path = self.anchor_positive_pairs[idx]
        negative_path = np.random.choice(self.negative_paths)

        # Cargar imágenes
        anchor = Image.open(anchor_path).convert('RGB')
        positive = Image.open(positive_path).convert('RGB')
        negative = Image.open(negative_path).convert('RGB')

        if self.transform:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)

        return anchor, positive, negative

# =========================
# 2️⃣ Transformaciones
# =========================
def get_transforms():
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        # Normalización opcional si quieres usar pretrained ResNet
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

# =========================
# 3️⃣ Modelo: ResNet Embedding
# =========================
class ResNetEmbedding(nn.Module):
    """
    Backbone ResNet con embedding vector L2-normalizado
    """
    def __init__(self, backbone='resnet18', embedding_dim=128, pretrained=True):
        super(ResNetEmbedding, self).__init__()
        if backbone == 'resnet18':
            self.resnet = models.resnet18(pretrained=pretrained)
        elif backbone == 'resnet50':
            self.resnet = models.resnet50(pretrained=pretrained)
        else:
            raise ValueError("Backbone no soportado")

        # Reemplazar la capa final por embedding
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, embedding_dim)

    def forward(self, x):
        embedding = self.resnet(x)
        # L2 normalizar embeddings para Triplet Loss y comparación
        return F.normalize(embedding, p=2, dim=1)

# =========================
# 4️⃣ Función de entrenamiento
# =========================
def train_triplet(model, dataloader, device, lr=1e-4, epochs=20, margin=1.0, save_path='resnet_triplet_best.pth'):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.TripletMarginLoss(margin=margin, p=2)
    best_loss = float('inf')

    model.to(device)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for anchor, positive, negative in dataloader:
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            optimizer.zero_grad()
            emb_anchor = model(anchor)
            emb_positive = model(positive)
            emb_negative = model(negative)
            loss = criterion(emb_anchor, emb_positive, emb_negative)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * anchor.size(0)

        epoch_loss = running_loss / len(dataloader.dataset)
        print(f"Epoch {epoch+1}/{epochs}, Triplet Loss: {epoch_loss:.6f}")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), save_path)
            print(f"Modelo guardado con mejor loss: {best_loss:.6f}")

    print("Entrenamiento finalizado.")

# =========================
# 5️⃣ Funciones de evaluación
# =========================
def calculate_embedding_distance(anchor_path, model, negative_mean_embedding, device):
    """
    Devuelve la distancia del embedding del patch al embedding medio de negativos
    """
    model.eval()
    transform = get_transforms()
    img = Image.open(anchor_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model(img_tensor)
    # Distancia euclídea al embedding medio de negativos
    dist = torch.norm(emb - negative_mean_embedding, p=2).item()
    return dist

def compute_negative_mean_embedding(model, negative_paths, device):
    """
    Calcular embedding promedio de todos los patches negativos (base de comparación)
    """
    model.eval()
    transform = get_transforms()
    embeddings = []
    with torch.no_grad():
        for path in negative_paths:
            img = Image.open(path).convert('RGB')
            tensor = transform(img).unsqueeze(0).to(device)
            emb = model(tensor)
            embeddings.append(emb)
    return torch.mean(torch.cat(embeddings, dim=0), dim=0, keepdim=True)

# =========================
# 6️⃣ Diagnóstico a nivel paciente
# =========================
def diagnose_patient_embeddings(pat_id, patches_root, model, negative_mean_embedding, tau, agg_pct=0.05, device='cpu'):
    """
    Diagnóstico por agregación usando embeddings y distancia al negativo promedio
    """
    pat_sections = glob.glob(os.path.join(patches_root, f"{pat_id}_*"))
    all_patch_files = []
    for section in pat_sections:
        all_patch_files.extend(glob.glob(os.path.join(section, '*.png')))

    if not all_patch_files:
        return None

    positive_patches = 0
    total_patches = len(all_patch_files)

    for patch_path in all_patch_files:
        dist = calculate_embedding_distance(patch_path, model, negative_mean_embedding, device)
        if dist >= tau:
            positive_patches += 1

    positive_ratio = positive_patches / total_patches
    prediction = 1 if positive_ratio >= agg_pct else 0
    return prediction

# =========================
# 7️⃣ Ejemplo de uso
# =========================
if __name__ == '__main__':
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {DEVICE}")

    # Rutas de ejemplo (ajustar según tu estructura)
    NEGATIVA_FILE = 'negativa_patients.txt'
    POSITIVA_FILE = 'positiva_patients.txt'  # Opcional para entrenamiento
    PATCHES_ROOT = '/export/fhome/maed/HelicoDataSet/CrossValidation/Cropped/'

    # Cargar lista de paths
    with open(NEGATIVA_FILE, 'r') as f:
        negativa_patients = [line.strip() for line in f if line.strip()]
    negative_paths = []
    for pat_id in negativa_patients:
        for folder in glob.glob(os.path.join(PATCHES_ROOT, f"{pat_id}_*")):
            negative_paths.extend(glob.glob(os.path.join(folder, '*.png')))

    # Similar para positivos
    with open(POSITIVA_FILE, 'r') as f:
        positiva_patients = [line.strip() for line in f if line.strip()]
    positive_paths = []
    for pat_id in positiva_patients:
        for folder in glob.glob(os.path.join(PATCHES_ROOT, f"{pat_id}_*")):
            positive_paths.extend(glob.glob(os.path.join(folder, '*.png')))

    # Dataset y DataLoader
    transforms_ = get_transforms()
    dataset = TripletPatchDataset(positive_paths, negative_paths, transform=transforms_)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

    # Modelo
    model = ResNetEmbedding(backbone='resnet18', embedding_dim=128, pretrained=True)

    # Entrenamiento
    train_triplet(model, dataloader, DEVICE, lr=1e-4, epochs=20, margin=1.0, save_path='resnet_triplet_best.pth')

    # Calcular embedding promedio de negativos
    negative_mean_emb = compute_negative_mean_embedding(model, negative_paths, DEVICE)

    # Diagnóstico ejemplo
    test_pat_id = negativa_patients[0]
    prediction = diagnose_patient_embeddings(test_pat_id, PATCHES_ROOT, model, negative_mean_emb, tau=0.5, agg_pct=0.05, device=DEVICE)
    print(f"Predicción paciente {test_pat_id}: {prediction}")
