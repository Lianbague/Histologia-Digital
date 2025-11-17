# patient_diagnosis_ae_kfold.py
# -*- coding: utf-8 -*-
import os
import sys
import glob
import pandas as pd
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import torch.nn as nn
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score, f1_score

# Assegura't que ae_models.py existeix al mateix directori
from ae_models import AutoEncoderCNN, AEConfigs 

# --- PARAMETRES CRITICS (TUNING 7: PRIORITAT SENSINILITAT) ---
OPTIMAL_TAU = 0.000350 
AGGREGATION_THRESHOLD_PCT = 0.05 # 5.0%
NUM_FOLDS = 5 # 5 Particions

# Model de L1 Loss
MODEL_SAVE_PATH = 'autoencoder_negativa_best_L1Loss.pth' 
CSV_DIAGNOSIS_PATH = '/export/fhome/maed/HelicoDataSet/PatientDiagnosis.csv'
CV_PATCHES_ROOT = '/export/fhome/maed/HelicoDataSet/CrossValidation/Cropped'
HOLDOUT_PATCHES_ROOT = '/export/fhome/maed/HelicoDataSet/HoldOut'

# ... (Funcions Auxiliars: get_eval_transforms, calculate_reconstruction_error, load_model - Es mantenen) ...

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
        # Cal utilitzar la L2 Loss com a metrica de decisio, ja que historicament es el que s'utilitza.
        # Encara que l'entrenament sigui L1, la metrica de l'error per a l'anomalia es L2.
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
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model


def diagnose_patient(pat_id, patches_root, model, device):
    """ Realitza la diagnosi d'un sol pacient per agregacio. """
    pat_sections = glob.glob(os.path.join(patches_root, f"{pat_id}_*"))
    all_patch_files = []
    for section in pat_sections:
        all_patch_files.extend(glob.glob(os.path.join(section, '*.png')))
        
    if not all_patch_files:
        return None # No hi ha pedacos

    positive_patches = 0
    total_patches = len(all_patch_files)
    
    for patch_path in all_patch_files:
        error = calculate_reconstruction_error(patch_path, model, device)
        if error is not None and error > OPTIMAL_TAU:
            positive_patches += 1
            
    positive_ratio = positive_patches / total_patches
    prediction = 1 if positive_ratio >= AGGREGATION_THRESHOLD_PCT else 0
    return prediction


def run_assessment(model, device, df_patients, assessment_name, patches_root):
    """ Executa la diagnosi per a un grup de pacients i avalua el rendiment. """
    
    patient_results = {'PatID': [], 'GT_Binary': [], 'Prediction': []}
    
    print(f"\n--- Iniciant Avaluacio: {assessment_name} ({len(df_patients)} pacients) ---")
    
    for index, row in df_patients.iterrows():
        pat_id = row['CODI']
        sys.stdout.write(f"\rProcessant pacient: {pat_id}...")
        sys.stdout.flush()
        
        prediction = diagnose_patient(pat_id, patches_root, model, device)
        
        if prediction is not None:
            patient_results['PatID'].append(pat_id)
            patient_results['GT_Binary'].append(row['GT_Binary'])
            patient_results['Prediction'].append(prediction)

    final_df = pd.DataFrame(patient_results)
    if final_df.empty:
        print(f"Advertencia: No s'ha pogut avaluar cap pacient en {assessment_name}.")
        return None

    accuracy = accuracy_score(final_df['GT_Binary'], final_df['Prediction'])
    recall = recall_score(final_df['GT_Binary'], final_df['Prediction'], zero_division=0)
    precision = precision_score(final_df['GT_Binary'], final_df['Prediction'], zero_division=0)
    f1 = f1_score(final_df['GT_Binary'], final_df['Prediction'], zero_division=0)
    
    # La matriu es calcula al final de la llista de pacients
    tn, fp, fn, tp = confusion_matrix(final_df['GT_Binary'], final_df['Prediction'], labels=[0, 1]).ravel() 
    
    print(f"\n--- RESULTATS FINALS: {assessment_name} ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f} | Recall (Sensibilitat): {recall:.4f} | F1-Score: {f1:.4f}")
    print("Matriu de Confusio:")
    print(f"TN (Negatiu Correcte): {tn}, FP (Fals Positiu): {fp}")
    print(f"FN (Fals Negatiu): {fn}, TP (Positiu Correcte): {tp}")
    
    return {'accuracy': accuracy, 'recall': recall, 'precision': precision, 'f1': f1}


def main_assessment():
    
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Utilitzant dispositiu: {DEVICE}")

    # 1. Carrega de Diagnoses Globals
    df_diagnosis = pd.read_csv(CSV_DIAGNOSIS_PATH)
    df_diagnosis['GT_Binary'] = df_diagnosis['DENSITAT'].apply(lambda x: 0 if x == 'NEGATIVA' else 1)
    
    # 2. Separacio dels Sets (CV vs HoldOut)
    cv_pat_ids = set([os.path.basename(f).split('_')[0] for f in glob.glob(os.path.join(CV_PATCHES_ROOT, '*'))])
    holdout_pat_ids = set([os.path.basename(f).split('_')[0] for f in glob.glob(os.path.join(HOLDOUT_PATCHES_ROOT, '*'))])
    
    df_cv = df_diagnosis[df_diagnosis['CODI'].isin(cv_pat_ids)].reset_index(drop=True)
    df_holdout = df_diagnosis[df_diagnosis['CODI'].isin(holdout_pat_ids)].reset_index(drop=True)
    
    # Excloure els pacients de CV que per error podrien estar a HoldOut (i viceversa)
    df_holdout = df_holdout[~df_holdout['CODI'].isin(cv_pat_ids)]

    print(f"Pacients per a CV: {len(df_cv)}")
    print(f"Pacients per a Generalitzacio (HoldOut): {len(df_holdout)}")

    # Carreguem l'unic model entrenat 
    if not os.path.exists(MODEL_SAVE_PATH):
        print(f"ERROR: Model file not found at {MODEL_SAVE_PATH}. Entrena l'AE amb L1 Loss primer.")
        sys.exit(1)
        
    model = load_model(DEVICE)

    # 3. Assessment of System 1: KFold Schemes (Cross-Validation)
    print("\n\n#####################################################")
    print("## Avaluacio KFold (Cross-Validation) a nivell de Pacient ##")
    print("#####################################################")
    
    kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
    cv_metrics = {'accuracy': [], 'recall': [], 'precision': [], 'f1': []}
    
    for fold, (train_index, test_index) in enumerate(kf.split(df_cv)):
        # Avaluem l'AE fixe nomes al TEST set de pacients del CV
        df_test_fold = df_cv.iloc[test_index]
        
        # Saltem els missatges de progres per velocitat en KFold
        sys.stdout.write(f"\rExecutant Fold {fold+1}/{NUM_FOLDS}...")
        sys.stdout.flush()

        fold_results = run_assessment(model, DEVICE, df_test_fold, 
                                      f"CV Fold {fold+1}/{NUM_FOLDS}", CV_PATCHES_ROOT)
        
        if fold_results:
            for key in cv_metrics:
                cv_metrics[key].append(fold_results[key])
    
    # Resultats Finals del KFold
    print("\n=========================================================================")
    print(f"RESULTATS FINALS K-FOLD ({NUM_FOLDS} FOLDS) SISTEMA 1")
    for key, values in cv_metrics.items():
        print(f"Media {key.capitalize()}: {np.mean(values):.4f} (+/- {np.std(values):.4f})")
    print("=========================================================================")


    # 4. Assessment of System 1: Generalization Levels (HoldOut)
    print("\n\n##############################################################")
    print("## Avaluacio de Generalitzacio (HoldOut) a nivell de Pacient ##")
    print("##############################################################")
    
    run_assessment(model, DEVICE, df_holdout, "HoldOut Set", HOLDOUT_PATCHES_ROOT)


if __name__ == '__main__':
    main_assessment()
