# patient_diagnosis_ae_kfold_V2.py
# -*- coding: utf-8 -*-
import os
import sys
import glob
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score, f1_score, roc_curve

from AE_train_1.ae_models import AutoEncoderCNN, AEConfigs 

NUM_FOLDS = 5 # 5 Particiones
KFOLD_RANDOM_STATE = 42

CSV_DIAGNOSIS_PATH = '/export/fhome/maed/HelicoDataSet/PatientDiagnosis.csv'
CV_PATCHES_ROOT = '/export/fhome/maed/HelicoDataSet/CrossValidation/Cropped'
HOLDOUT_PATCHES_ROOT = '/export/fhome/maed/HelicoDataSet/HoldOut'

# Paths per a la clasificacio de parxes
ANNOTATED_CSV_PATH = '/fhome/maed03/data_preprocessing_0/threshold_set_balanced.csv' 
ANNOTATED_PATCHES_ROOT = '/export/fhome/maed/HelicoDataSet/CrossValidation/Annotated'

# Hiperparametres d'Entrenament
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
NUM_EPOCHS = 15


# Entrenament
class PatchDataset(Dataset):
    """Dataset per carregar nomes els paths de les imatges sanes."""
    def __init__(self, file_paths, transform=None):
        self.file_paths = file_paths
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB') 
            if self.transform:
                image = self.transform(image)
            # Per a un AE, l'entrada i l'objectiu son el mateix
            return image, image
        except Exception as e:
            # Retorna None si una imatge esta corrupta
            print(f"WARN: No s'ha pogut carregar {img_path}. Error: {e}")
            return None, None

def collate_fn(batch):
    """Funcio per filtrar imatges corruptes (que son None)."""
    batch = list(filter(lambda x: x[0] is not None, batch))
    if not batch:
        return torch.Tensor(), torch.Tensor()
    return torch.utils.data.dataloader.default_collate(batch)

def get_train_transforms():
    """ Transformacions per a l'entrenament, incloent normalitzacio. """
    return transforms.Compose([
        transforms.Resize((256, 256)), 
        transforms.ToTensor(),       
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ])

def train_new_ae(train_pat_ids, df_diagnosis, fold_num, device):
    """
    FASE 1 (K-FOLD): Entrena un nou model AE des de zero.
    Utilitza nomes els pedacos SANS (NEGATIVA) dels pacients de TRAIN.
    """
    print(f"  [Fase 1 - Fold {fold_num}] Iniciant entrenament...")
    
    # 1. Trobar els pacients sans (NEGATIVA) dels IDs de train
    df_train_patients = df_diagnosis[df_diagnosis['CODI'].isin(train_pat_ids)]
    df_sana = df_train_patients[df_train_patients['DENSITAT'] == 'NEGATIVA']
    sana_pat_ids = df_sana['CODI'].tolist()
    
    # 2. Recollir tots els paths dels pedaços d'aquests pacients
    all_patch_paths = []
    for pat_id in sana_pat_ids:
        pat_sections = glob.glob(os.path.join(CV_PATCHES_ROOT, f"{pat_id}_*"))
        for section in pat_sections:
            all_patch_paths.extend(glob.glob(os.path.join(section, '*.png')))
            
    if not all_patch_paths:
        print("ERROR: No s'han trobat pedaços d'entrenament sans per a aquest fold.")
        return None

    print(f"  [Fase 1 - Fold {fold_num}] {len(all_patch_paths)} pedacos sans per entrenar.")
    
    # 3. Preparar DataLoader
    train_transforms = get_train_transforms()
    dataset = PatchDataset(all_patch_paths, transform=train_transforms)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, 
                            num_workers=4, pin_memory=True, collate_fn=collate_fn)
    
    # 4. Configurar i entrenar el model
    config = AEConfigs(config_id='1', input_channels=3)
    model = AutoEncoderCNN(
        inputmodule_paramsEnc=config.inputmodule_paramsEnc, 
        net_paramsEnc=config.net_paramsEnc, 
        inputmodule_paramsDec=config.inputmodule_paramsDec, 
        net_paramsDec=config.net_paramsDec
    )
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.L1Loss() # Utilitzant L1Loss (MAE) com al teu model original
    
    best_loss = float('inf')
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        
        for i, (inputs, targets) in enumerate(dataloader):
            if inputs.shape[0] == 0: continue # Ometre batches buits (d'imatges corruptes)
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            reconstructions = model(inputs)
            loss = criterion(reconstructions, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            
        epoch_loss = running_loss / len(dataset)
        # Guardar el millor model d'aquest fold
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            # No guardem a disc per estalviar espai, nomes mantenim el model en memoria
            # (Si vols guardar-lo, fes-ho amb un path unic: ex. f'AE_fold_{fold_num}.pth')

        if (epoch + 1) % 10 == 0: # Imprimir progres
            print(f"    Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {epoch_loss:.6f}")

    print(f"  [Fase 1 - Fold {fold_num}] Entrenament finalitzat. Millor Loss: {best_loss:.6f}")
    return model.eval() # Retornar el model entrenat en mode avaluacio


# Clasificacio

def get_eval_transforms():
    """ Transformacions per a avaluacio/inferencia (SENSE normalitzacio). """
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

def calculate_reconstruction_error(image_path, model, device):
    """ Calcula l'error de reconstruccio (MSE/L2) d'una sola imatge. """
    if not os.path.exists(image_path):
        return None
        
    transforms_eval = get_eval_transforms()
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception:
        return None # Imatge corrupta

    input_tensor = transforms_eval(image).unsqueeze(0).to(device) 
    model.eval()
    with torch.no_grad():
        reconstruction = model(input_tensor)
        # La metrica d'error per anomalia es MSE (L2), encara que entrenem amb L1
        l_red = nn.MSELoss(reduction='none')(reconstruction, input_tensor).mean(dim=[1, 2, 3])
        return l_red.item()

def find_optimal_patch_threshold(model, train_pat_ids, device):
    """
    FASE 2 (K-FOLD): Troba el llindar de pedacos (Threshold) optim.
    Utilitza nomes el set de calibracio (Annotated) dels pacients de TRAIN.
    """
    print("  [Fase 2] Calibrant llindar de pedacos (Tau)...")
    
    try:
        df_annotated = pd.read_csv(ANNOTATED_CSV_PATH)
    except Exception as e:
        print(f"ERROR: No s'ha pogut carregar {ANNOTATED_CSV_PATH}. Error: {e}")
        return None # Error fatal
        
    # Extreure PatID del 'Pat_Section' (ex: 'B22-101_0' -> 'B22-101')
    df_annotated['PatID'] = df_annotated['Pat_Section'].apply(lambda x: x.split('_')[0])
    
    # Filtrar nomes els pedacos dels pacients de TRAIN
    df_train_annotated = df_annotated[df_annotated['PatID'].isin(train_pat_ids)]
    
    if df_train_annotated.empty:
        print("ERROR: No s'han trobat pedacos anotats per als pacients d'entrenament.")
        return None

    errors = []
    labels = []
    
    for _, row in df_train_annotated.iterrows():
        # Reconstruir el path del pedac (com a generate_ROC.py)
        pat_section_folder = row['Pat_Section']
        window_id = row['Window_ID']
        num_part = int(re.match(r"(\d+)", str(window_id)).group(1))
        image_filename = f"{num_part:05d}.png" # Assumim que no hi ha Augmentacio
        image_path = os.path.join(ANNOTATED_PATCHES_ROOT, pat_section_folder, image_filename)

        error = calculate_reconstruction_error(image_path, model, device)
        if error is not None:
            errors.append(error)
            labels.append(1 if row['Presence'] == 1 else 0)
            
    if not errors:
        print("ERROR: No s'ha pogut calcular l'error de cap pedac anotat.")
        return None

    # Calcular ROC per trobar el millor llindar
    fpr, tpr, thresholds = roc_curve(labels, errors)
    # Estrategia: Maximitzar (TPR - FPR) (punt mes proper a cantonada 0,1)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_tau = thresholds[optimal_idx]
    
    return optimal_tau

def get_patient_score(pat_id, patches_root, model, device, optimal_tau):
    """
    Calcula el "score" d'un pacient: el percentatge de pedacos positius.
    """
    pat_sections = glob.glob(os.path.join(patches_root, f"{pat_id}_*"))
    all_patch_files = []
    for section in pat_sections:
        all_patch_files.extend(glob.glob(os.path.join(section, '*.png')))
        
    if not all_patch_files:
        return None # No hi ha pedacos

    positive_patches = 0
    total_patches = 0
    
    for patch_path in all_patch_files:
        error = calculate_reconstruction_error(patch_path, model, device)
        if error is not None:
            total_patches += 1
            if error > optimal_tau:
                positive_patches += 1
                
    if total_patches == 0:
        return None # No s'ha pogut processar cap pedac

    positive_ratio = positive_patches / total_patches
    return positive_ratio

def find_optimal_patient_threshold(model, optimal_tau, train_pat_ids, df_train_fold, device):
    """
    FASE 3A (K-FOLD): Troba el llindar d'agregacio (%) optim.
    Utilitza els "scores" de pacients generats sobre el set de TRAIN.
    """
    print("  [Fase 3A] Calibrant llindar de pacient (Agregacio)...")
    
    patient_scores = []
    patient_labels = []
    
    for pat_id in train_pat_ids:
        score = get_patient_score(pat_id, CV_PATCHES_ROOT, model, device, optimal_tau)
        
        if score is not None:
            patient_scores.append(score)
            label = df_train_fold[df_train_fold['CODI'] == pat_id]['GT_Binary'].values[0]
            patient_labels.append(label)
            
    if not patient_scores:
        print("ERROR: No s'ha pogut calcular l'score de cap pacient d'entrenament.")
        return None

    # Calcular ROC sobre els scores (percentatges) dels pacients
    fpr, tpr, thresholds = roc_curve(patient_labels, patient_scores)
    # Estrategia: Maximitzar (TPR - FPR)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_agg_thresh = thresholds[optimal_idx]
    
    # Assegurar-se que el llindar no es 0.0 o 1.0 (casos extrems)
    if optimal_agg_thresh <= 0.0: optimal_agg_thresh = 1e-6
    if optimal_agg_thresh >= 1.0: optimal_agg_thresh = 1.0 - 1e-6
    
    return optimal_agg_thresh


# Avaluacio
def run_assessment(model, optimal_tau, agg_thresh, df_patients, assessment_name, patches_root, device):
    """ 
    FASE 3B (K-FOLD): Executa la diagnosi per a un grup de pacients (TEST) 
    i avalua el rendiment usant els llindars trobats.
    """
    
    patient_results = {'PatID': [], 'GT_Binary': [], 'Prediction': []}
    
    print(f"  [Fase 3B] Testejant {assessment_name} ({len(df_patients)} pacients)...")
    
    for index, row in df_patients.iterrows():
        pat_id = row['CODI']
        
        # Obtenir el % de pedacos positius
        score = get_patient_score(pat_id, patches_root, model, device, optimal_tau)
        
        if score is not None:
            # Aplicar el llindar d'agregacio (Fase 3A)
            prediction = 1 if score >= agg_thresh else 0
            
            patient_results['PatID'].append(pat_id)
            patient_results['GT_Binary'].append(row['GT_Binary'])
            patient_results['Prediction'].append(prediction)
        else:
            print(f"WARN: No s'ha pogut processar el pacient {pat_id} (sense pedacos).")

    final_df = pd.DataFrame(patient_results)
    if final_df.empty:
        print(f"Advertencia: No s'ha pogut avaluar cap pacient en {assessment_name}.")
        return None

    # Calcular metriques
    accuracy = accuracy_score(final_df['GT_Binary'], final_df['Prediction'])
    recall = recall_score(final_df['GT_Binary'], final_df['Prediction'], zero_division=0)
    precision = precision_score(final_df['GT_Binary'], final_df['Prediction'], zero_division=0)
    f1 = f1_score(final_df['GT_Binary'], final_df['Prediction'], zero_division=0)
    
    # Matriu de confusio
    tn, fp, fn, tp = confusion_matrix(final_df['GT_Binary'], final_df['Prediction'], labels=[0, 1]).ravel() 
    
    print(f"\n  --- RESULTATS: {assessment_name} ---")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f} | Recall (Sensibilitat): {recall:.4f} | F1-Score: {f1:.4f}")
    print(f"  TN (Negatiu Correcte): {tn}, FP (Fals Positiu): {fp}, FN (Fals Negatiu): {fn}, TP (Positiu Correcte): {tp}")
    
    return {'accuracy': accuracy, 'recall': recall, 'precision': precision, 'f1': f1}


# Bucle principal K-FOLD
def main_assessment():
    
    print(f"Utilitzant dispositiu: {DEVICE}")

    # 1. Carrega de Diagnoses Globals
    try:
        df_diagnosis = pd.read_csv(CSV_DIAGNOSIS_PATH)
        df_diagnosis['GT_Binary'] = df_diagnosis['DENSITAT'].apply(lambda x: 0 if x == 'NEGATIVA' else 1)
    except Exception as e:
        print(f"ERROR: No s'ha pogut carregar {CSV_DIAGNOSIS_PATH}. Error: {e}")
        sys.exit(1)
        
    # 2. Separacio dels Sets (CV vs HoldOut)
    cv_pat_ids = set([os.path.basename(f).split('_')[0] for f in glob.glob(os.path.join(CV_PATCHES_ROOT, '*'))])
    holdout_pat_ids = set([os.path.basename(f).split('_')[0] for f in glob.glob(os.path.join(HOLDOUT_PATCHES_ROOT, '*'))])
    
    df_cv = df_diagnosis[df_diagnosis['CODI'].isin(cv_pat_ids)].reset_index(drop=True)
    df_holdout = df_diagnosis[df_diagnosis['CODI'].isin(holdout_pat_ids)]
    # Excloure els pacients de CV que per error podrien estar a HoldOut (i viceversa)
    df_holdout = df_holdout[~df_holdout['CODI'].isin(cv_pat_ids)].reset_index(drop=True)

    print(f"Pacients totals per a CV: {len(df_cv)}")
    print(f"Pacients totals per a Generalitzacio (HoldOut): {len(df_holdout)}")

    # 3. Assessment of System 1: KFold Schemes (Cross-Validation)
    print("\n\n#####################################################")
    print("## Avaluacio K-FOLD COMPLETA (Cross-Validation) ##")
    print("#####################################################")
    
    kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=KFOLD_RANDOM_STATE)
    cv_metrics = {'accuracy': [], 'recall': [], 'precision': [], 'f1': []}
    
    for fold, (train_index, test_index) in enumerate(kf.split(df_cv)):
        
        print(f"\n--- INICIANT FOLD {fold+1}/{NUM_FOLDS} ---")
        
        df_train_fold = df_cv.iloc[train_index]
        df_test_fold = df_cv.iloc[test_index]
        train_ids = df_train_fold['CODI'].tolist()
        
        # Entrenar AE 
        model_k = train_new_ae(train_ids, df_diagnosis, fold+1, DEVICE)
        if model_k is None:
            print(f"ERROR: No s'ha pogut entrenar el model al Fold {fold+1}. Saltant...")
            continue

        tau_k = find_optimal_patch_threshold(model_k, train_ids, DEVICE)
        if tau_k is None:
            print(f"ERROR: No s'ha pogut calibrar Tau al Fold {fold+1}. Saltant...")
            continue
        print(f"  [Fase 2 - Fold {fold+1}] Llindar de Pedac (Tau) trobat: {tau_k:.6f}")

        
        agg_thresh_k = find_optimal_patient_threshold(model_k, tau_k, train_ids, df_train_fold, DEVICE)
        if agg_thresh_k is None:
            print(f"ERROR: No s'ha pogut calibrar AggThresh al Fold {fold+1}. Saltant...")
            continue
        print(f"  [Fase 3A - Fold {fold+1}] Llindar de Pacient (Agg) trobat: {agg_thresh_k:.6f}")

        # Test
        fold_results = run_assessment(model_k, tau_k, agg_thresh_k, df_test_fold, 
                                      f"CV Test Fold {fold+1}", CV_PATCHES_ROOT, DEVICE)
        
        if fold_results:
            for key in cv_metrics:
                cv_metrics[key].append(fold_results[key])
        
        # Netejar memoria de la GPU
        del model_k
        torch.cuda.empty_cache()

    # Resultats Finals del KFold
    print("\n=========================================================================")
    print(f"RESULTATS FINALS K-FOLD ({NUM_FOLDS} FOLDS) SISTEMA 1")
    for key, values in cv_metrics.items():
        if values:
            print(f"Media {key.capitalize()}: {np.mean(values):.4f} (+/- {np.std(values):.4f})")
        else:
            print(f"No s'han pogut calcular metriques per a {key}")
    print("=========================================================================")

    # 4. Avaluacio Final (HoldOut) - (Opcional, pero bona practica)
    
    print("\n\n##############################################################")
    print("## Avaluacio de Generalitzacio (HoldOut) ##")
    print("##############################################################")
    
    print("Entrenant model final amb TOTES les dades de CV...")
    all_cv_ids = df_cv['CODI'].tolist()
    
    model_final = train_new_ae(all_cv_ids, df_diagnosis, "FINAL", DEVICE)
    
    if model_final:

        tau_final = find_optimal_patch_threshold(model_final, all_cv_ids, DEVICE)
        print(f"Llindar de Pedac Final (Tau) trobat: {tau_final:.6f}")
        
        agg_thresh_final = find_optimal_patient_threshold(model_final, tau_final, all_cv_ids, df_cv, DEVICE)
        print(f"Llindar de Pacient Final (Agg) trobat: {agg_thresh_final:.6f}")

        if tau_final and agg_thresh_final:
            print("\nTestejant el model final sobre el HoldOut Set...")
            run_assessment(model_final, tau_final, agg_thresh_final, df_holdout, 
                           "HoldOut Set", HOLDOUT_PATCHES_ROOT, DEVICE)
        else:
            print("ERROR: No s'han pogut trobar els llindars finals. No es pot avaluar el HoldOut.")
    else:
        print("ERROR: No s'ha pogut entrenar el model final. No es pot avaluar el HoldOut.")

if __name__ == '__main__':
    main_assessment()