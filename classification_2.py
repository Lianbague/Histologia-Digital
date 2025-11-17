# -*- coding: utf-8 -*-
import os
import glob
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
from ae_models import AutoEncoderCNN, AEConfigs

# --- Parámetros ---
NEGATIVA_FILE = 'nueva_negativa_patients.txt'
PATCHES_ROOT = '/export/fhome/maed/HelicoDataSet/CrossValidation/Cropped/'
MODEL_SAVE_PATH = 'autoencoder_negativa_best_L1Loss.pth'

AGGREGATION_THRESHOLD_PCT = 0.05
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --- Funciones ---
def get_eval_transforms():
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

def load_model(device):
    config = AEConfigs(config_id='1', input_channels=3)
    model = AutoEncoderCNN(
        net_paramsEnc=config.net_paramsEnc, 
        inputmodule_paramsDec=config.inputmodule_paramsDec, 
        net_paramsDec=config.net_paramsDec
    )
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model

def extract_embedding(model, image_path):
    if not os.path.exists(image_path):
        return None
    transform = get_eval_transforms()
    try:
        image = Image.open(image_path).convert('RGB')
    except:
        return None
    x = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        emb = model.encoder(x)
        emb_flat = emb.view(emb.size(0), -1)
    return emb_flat.cpu().numpy()[0]

def build_negative_center(neg_patients):
    embeddings = []
    for pat_id in neg_patients:
        pat_folders = glob.glob(os.path.join(PATCHES_ROOT, f"{pat_id}_*"))
        for folder in pat_folders:
            patches = glob.glob(os.path.join(folder, '*.png'))
            for patch in patches:
                emb = extract_embedding(model, patch)
                if emb is not None:
                    embeddings.append(emb)
    embeddings = np.array(embeddings)
    center = np.mean(embeddings, axis=0)
    # Calcular distancia percentil 95 como umbral inicial
    global OPTIMAL_TAU
    dists = np.linalg.norm(embeddings - center, axis=1)
    OPTIMAL_TAU = np.percentile(dists, 95)
    print(f"Centro de negativos calculado. Tau inicial (percentil 95): {OPTIMAL_TAU:.6f}")
    return center

def diagnose_patient_embeddings(pat_id, patches_root, model, center, tau, agg_pct):
    pat_folders = glob.glob(os.path.join(patches_root, f"{pat_id}_*"))
    all_patch_files = []
    for folder in pat_folders:
        all_patch_files.extend(glob.glob(os.path.join(folder, '*.png')))
    if not all_patch_files:
        return None, 0, 0  # No hay patches
    positive_count = 0
    for patch in all_patch_files:
        emb = extract_embedding(model, patch)
        if emb is not None:
            dist = np.linalg.norm(emb - center)
            if dist > tau:
                positive_count += 1
    ratio = positive_count / len(all_patch_files)
    prediction = 1 if ratio >= agg_pct else 0
    return prediction, positive_count, len(all_patch_files)

# --- Main ---
if __name__ == '__main__':
    print(f"Usando dispositivo: {DEVICE}")
    if not os.path.exists(MODEL_SAVE_PATH):
        print(f"ERROR: No se encontró el modelo {MODEL_SAVE_PATH}")
        exit(1)

    model = load_model(DEVICE)

    # --- 1. Cargar lista de pacientes negativos ---
    with open(NEGATIVA_FILE, 'r') as f:
        neg_patients = [line.strip() for line in f if line.strip()]
    print(f"Pacientes negativos cargados: {neg_patients}")

    # --- 2. Construir centro de negativos ---
    negative_center = build_negative_center(neg_patients)

    # --- 3. Diagnóstico de todos los pacientes negativos ---
    print("\nDiagnóstico pacientes negativos (control):")
    for pat_id in neg_patients:
        pred, pos_count, total_patches = diagnose_patient_embeddings(
            pat_id, PATCHES_ROOT, model, negative_center, tau=OPTIMAL_TAU, agg_pct=AGGREGATION_THRESHOLD_PCT
        )
        if pred is None:
            print(f"Paciente {pat_id}: No se encontraron patches.")
        else:
            print(f"Paciente {pat_id}: Predicción={pred} | Patches positivos={pos_count}/{total_patches} | Tau={OPTIMAL_TAU:.6f}")
