# patient_diagnosis_ae.py
# -*- coding: utf-8 -*-
import os
import sys
import glob
import pandas as pd
import torch
from torchvision import transforms
from PIL import Image
import torch.nn as nn
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score

# Assegura't que ae_models.py existeix al mateix directori
from ae_models import AutoEncoderCNN, AEConfigs 

# --- PARAMETRES CRITICS ---
OPTIMAL_TAU = 0.000350 # <-- LLINDAR OPTIM TROBAT AL PAS 4 (F1-score)
# Percentatge minim de pedacos malalts per classificar el pacient com a POSITIU.
# 0.05 es 5%.
AGGREGATION_THRESHOLD_PCT = 0.05 

MODEL_SAVE_PATH = 'autoencoder_negativa_best_L1Loss.pth'
CSV_DIAGNOSIS_PATH = '/export/fhome/maed/HelicoDataSet/PatientDiagnosis.csv'
PATCHES_ROOT = '/export/fhome/maed/HelicoDataSet/CrossValidation/Cropped'

# --- 1. Funcions Auxiliars ---

def get_eval_transforms():
    """ Usa les mateixes transformacions SENSE NORMALITZACIO que l'entrenament. """
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

def calculate_reconstruction_error(image_path, model, device):
    """ Calcula l'error de reconstruccio d'una sola imatge. """
    if not os.path.exists(image_path):
        return None
        
    transforms_eval = get_eval_transforms()
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception:
        return None

    input_tensor = transforms_eval(image).unsqueeze(0).to(device) 
    model.eval()
    with torch.no_grad():
        reconstruction = model(input_tensor)
        l_red = nn.MSELoss(reduction='none')(reconstruction, input_tensor).mean(dim=[1, 2, 3])
        return l_red.item()

def load_model(device):
    """ Carrega i retorna el model AE entrenat. """
    config = AEConfigs(config_id='1', input_channels=3)
    model = AutoEncoderCNN(
        net_paramsEnc=config.net_paramsEnc, 
        inputmodule_paramsDec=config.inputmodule_paramsDec, 
        net_paramsDec=config.net_paramsDec
    )
    # Cal carregar els pesos des del fitxer
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model

# --- 2. Logica Principal del Diagnostic ---

def run_patient_diagnosis(model, device):
    """ Llegeix les dades, realitza el diagnostic per agregacio i avalua el resultat. """
    
    # Carrega les diagnoses de pacients (NEGATIVA, BAIXA, ALTA)
    # CODI es Pat_ID, DENSITAT es la classificacio
    df_diagnosis = pd.read_csv(CSV_DIAGNOSIS_PATH) 
    
    # 1. Preparacio de les Dades
    # Filtrem nomes els pacients del set de CrossValidation (els que tenen carpeta a PATCHES_ROOT)
    all_pat_folders = glob.glob(os.path.join(PATCHES_ROOT, '*'))
    # El nom de la carpeta es PatID_Section# (p.ex., B22-01_1)
    cv_pat_ids = set([os.path.basename(f).split('_')[0] for f in all_pat_folders])
    
    df_cv = df_diagnosis[df_diagnosis['CODI'].isin(cv_pat_ids)].copy()
    
    # Mapeig de la veritat fonamental (Ground Truth):
    # NEGATIVA -> 0 (SA)
    # BAIXA/ALTA -> 1 (POSITIU/MALALT)
    df_cv['GT_Binary'] = df_cv['DENSITAT'].apply(lambda x: 0 if x == 'NEGATIVA' else 1)
    
    # Llistes per a l'avaluacio final
    patient_results = {'PatID': [], 'GT_Binary': [], 'Prediction': []}

    print(f"Iniciant diagnostic per a {len(df_cv)} pacients del set CrossValidation...")
    
    # 2. Iterar sobre cada pacient per fer el diagnostic (Aggregation)
    for index, row in df_cv.iterrows():
        pat_id = row['CODI']
        gt_label = row['GT_Binary']
        
        sys.stdout.write(f"\rProcessant pacient: {pat_id}...")
        sys.stdout.flush()
        
        # Trobar tots els pedacos d'aquest pacient (en totes les seccions, p.ex., B22-01_0, B22-01_1)
        pat_sections = glob.glob(os.path.join(PATCHES_ROOT, f"{pat_id}_*"))
        
        all_patch_files = []
        for section in pat_sections:
            # Afegim tots els pedacos .png de la seccio
            all_patch_files.extend(glob.glob(os.path.join(section, '*.png')))
            
        if not all_patch_files:
            continue # Salta el pacient si no troba pedacos
        
        # 3. Classificacio del Pedac i Agregacio
        positive_patches = 0
        total_patches = len(all_patch_files)
        
        for patch_path in all_patch_files:
            error = calculate_reconstruction_error(patch_path, model, device)
            
            if error is not None:
                # Classificacio a nivell de pedac: Malalt si Error > tau
                if error > OPTIMAL_TAU:
                    positive_patches += 1
        
        # 4. Diagnostic Final del Pacient (Aggregation)
        positive_ratio = positive_patches / total_patches
        
        if positive_ratio >= AGGREGATION_THRESHOLD_PCT:
            prediction = 1 # POSITIU (Malalt)
        else:
            prediction = 0 # NEGATIVA (SA)
            
        patient_results['PatID'].append(pat_id)
        patient_results['GT_Binary'].append(gt_label)
        patient_results['Prediction'].append(prediction)

    # 5. Avaluacio dels Resultats
    final_df = pd.DataFrame(patient_results)
    
    # Calculem les metriques sobre la diagnoses binaria (0/1)
    accuracy = accuracy_score(final_df['GT_Binary'], final_df['Prediction'])
    # labels=[0, 1] assegura que l'ordre es TN, FP, FN, TP
    tn, fp, fn, tp = confusion_matrix(final_df['GT_Binary'], final_df['Prediction'], labels=[0, 1]).ravel() 
    
    print("\n--- AVALUACIO DEL DIAGNOSTIC DE PACIENTS (SISTEMA 1) ---")
    print(f"Llindar d'Agregacio (Pct Pedacos Positius): {AGGREGATION_THRESHOLD_PCT * 100:.1f}%")
    print(f"Precisio (Accuracy): {accuracy:.4f}")
    print("\nMatriu de Confusio:")
    print(f"TN (Negatiu Correcte): {tn}, FP (Fals Positiu): {fp}")
    print(f"FN (Fals Negatiu): {fn}, TP (Positiu Correcte): {tp}")

if __name__ == '__main__':
    # Configura el dispositiu
    # Utilitzem 'cuda:0' si hi ha GPU, sino CPU.
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Utilitzant dispositiu: {DEVICE}")

    # Carrega el model
    if not os.path.exists(MODEL_SAVE_PATH):
        print(f"ERROR: Model file not found at {MODEL_SAVE_PATH}. Entrena l'AE primer.")
        sys.exit(1)

    model = load_model(DEVICE)
    
    # Executa la diagnosi
    run_patient_diagnosis(model, DEVICE)