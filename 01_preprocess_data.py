import pandas as pd
import os
import numpy as np
from PIL import Image
import pickle
from tqdm import tqdm

# --- CONFIGURACIÓ DE RUTES ---
DATA_ROOT = '/fhome/maed/HelicoDataSet' 
DIAGNOSIS_FILE = 'PatientDiagnosis.csv'
OUTPUT_FILE = 'sane_patches_data.pkl' # Fitxer on desarem les dades precarregades

# --- 1. IDENTIFICACIÓ DE PACIENTS SANS ---
print("1. Identificant pacients sans (NEGATIVA)...")
df_diagnosis = pd.read_csv(DIAGNOSIS_FILE)
sane_patients_df = df_diagnosis[df_diagnosis['DENSITAT'] == 'NEGATIVA']
AE_TRAIN_PATIENT_IDS = sane_patients_df['CODI'].tolist()

# --- 2. GENERACIÓ DE LA LLISTA DE RUTES ALS PATCHES ---
AE_TRAIN_PATHS = []
for pat_id in AE_TRAIN_PATIENT_IDS:
    # Hem d'explorar les subcarpetes PatID_Section#
    patient_dirs = [d for d in os.listdir(DATA_ROOT) if d.startswith(pat_id) and os.path.isdir(os.path.join(DATA_ROOT, d))]
    
    for p_dir in patient_dirs:
        full_path = os.path.join(DATA_ROOT, p_dir)
        patch_files = [os.path.join(full_path, f) for f in os.listdir(full_path) if f.endswith(('.png', '.jpg'))]
        AE_TRAIN_PATHS.extend(patch_files)

print(f"Total de patches a precargar: {len(AE_TRAIN_PATHS)}")

# --- 3. CÀRREGA I PREPROCESSAMENT ---
all_patches = []
patch_size = 256 # Assumint 256x256, 3 canals (RGB)

print("3. Carregant i preprocessant les imatges...")
for img_path in tqdm(AE_TRAIN_PATHS, desc="Precàrrega"):
    try:
        # Carrega la imatge, converteix a RGB
        image = Image.open(img_path).convert('RGB') 
        
        # Converteix a NumPy array i normalitza a [0, 1]
        # (PyTorch espera tensor C x H x W i valors de 0 a 1 per defecte)
        img_array = np.array(image, dtype=np.float32) / 255.0 
        
        # Les xarxes neuronals esperen (Channels, Height, Width)
        # Transposem de (H, W, C) a (C, H, W)
        img_array = img_array.transpose(2, 0, 1) 
        
        all_patches.append(img_array)
    except Exception as e:
        # Tractament d'errors per si hi ha algun fitxer corrupte
        # print(f"Error carregant {img_path}: {e}")
        continue

# Converteix la llista a un sol NumPy array
X_train = np.array(all_patches, dtype=np.float32)

print(f"Forma final de les dades precarregades: {X_train.shape}") # Ex: (N, 3, 256, 256)

# --- 4. DESA LES DADES PRECÀRREGADES ---
print(f"4. Desant les dades a {OUTPUT_FILE} usant pickle...")
with open(OUTPUT_FILE, 'wb') as f:
    pickle.dump(X_train, f)

print("Precàrrega de dades finalitzada.")