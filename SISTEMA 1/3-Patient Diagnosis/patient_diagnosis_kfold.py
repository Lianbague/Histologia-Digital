# -*- coding: utf-8 -*-
import os
import sys
import glob
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score
from sklearn.utils import resample
from scipy.interpolate import PchipInterpolator
import json
import random
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# --- IMPORTS DEL MODEL ---
from _1_AE_train.ae_models import AutoEncoderCNN, AEConfigs, VariationalAutoEncoderCNN

# ==========================================
# 1. CONFIGURACIÓ D'USUARI
# ==========================================

# --- OPCIONS PRINCIPALS ---
MODEL_TYPE = 'VAE'        # Opcions: 'AE' o 'VAE'
IMG_SIZE = 256           # Opcions: 128 o 256
ERROR_METRIC = 'P99'     # Opcions: 'MSE' o 'P99'

# --- LLINDAR MANUAL PER PATCH ---
# Aquest valor ve del teu anàlisi previ (Patch Classification)
MANUAL_PATCH_THRESHOLD = 0.1387

# --- ALTRES PARÀMETRES ---
LATENT_DIM = 128
MAX_PATCHES_PER_PATIENT = 500 # Limitar per velocitat (None = tots)
SMOOTH_ROC = True             # Suavitzar línies

# --- RUTES BASE ---
BASE_EXPORT = '/export/fhome/maed03'
BASE_DATASET = '/export/fhome/maed/HelicoDataSet'

FOLDS_JSON_PATH = os.path.join(BASE_EXPORT, '_2_fix_threshold/folds_distribution_10folds.json')
CSV_DIAGNOSIS_PATH = os.path.join(BASE_DATASET, 'PatientDiagnosis.csv')
CV_PATCHES_ROOT = os.path.join(BASE_DATASET, 'CrossValidation/Cropped')
HOLDOUT_PATCHES_ROOT = os.path.join(BASE_DATASET, 'HoldOut')

# --- CONFIG AUTOMÀTICA ---
if MODEL_TYPE == 'AE':
    filename = f'ae_L1Loss_{IMG_SIZE}.pth'
    MODEL_PATH = os.path.join(BASE_EXPORT, '_1_AE_train', filename)
else:
    filename = f'vae_{IMG_SIZE}.pth'
    MODEL_PATH = os.path.join(BASE_EXPORT, '_1_AE_train', filename)

output_folder_name = f"patient_diag_{MODEL_TYPE}_{IMG_SIZE}_{ERROR_METRIC}_500_3"
OUTPUT_DIR = os.path.join(BASE_EXPORT, '_4_patient_diagnosis_detailed', output_folder_name)
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(f"=== CONFIGURACIÓ ACTIVA ===")
print(f"Model: {MODEL_TYPE} | Mida: {IMG_SIZE} | Metric: {ERROR_METRIC} | Patches: {MAX_PATCHES_PER_PATIENT}")
print(f"Patch Threshold: {MANUAL_PATCH_THRESHOLD}")
print(f"Output: {OUTPUT_DIR}")
print("===========================")

# ==========================================
# 2. FUNCIONS AUXILIARS
# ==========================================

def rgb_to_hsv_h_channel_torch(image_tensor):
    r = image_tensor[:, 0, :, :]
    g = image_tensor[:, 1, :, :]
    b = image_tensor[:, 2, :, :]
    max_c, _ = torch.max(image_tensor, dim=1)
    min_c, _ = torch.min(image_tensor, dim=1)
    diff = max_c - min_c
    eps = 1e-7
    h = torch.zeros_like(max_c)
    mask_diff = (diff > 0)
    mask_r = (max_c == r) & mask_diff
    mask_g = (max_c == g) & mask_diff
    mask_b = (max_c == b) & mask_diff
    h[mask_r] = (g[mask_r] - b[mask_r]) / (diff[mask_r] + eps) % 6
    h[mask_g] = (b[mask_g] - r[mask_g]) / (diff[mask_g] + eps) + 2
    h[mask_b] = (r[mask_b] - g[mask_b]) / (diff[mask_b] + eps) + 4
    h = h / 6.0 
    return h 

def calculate_patient_ppp(pat_id, patches_root, model, device):
    """ Calcula Percentatge de Patchs Positius (PPP) """
    search_pattern = os.path.join(patches_root, f"{pat_id}*")
    patient_folders = glob.glob(search_pattern)
    all_image_paths = []
    for folder in patient_folders:
        if os.path.isdir(folder):
            all_image_paths.extend(glob.glob(os.path.join(folder, '*.png')))
            
    if not all_image_paths: return None

    if MAX_PATCHES_PER_PATIENT is not None and len(all_image_paths) > MAX_PATCHES_PER_PATIENT:
        paths_to_process = random.sample(all_image_paths, MAX_PATCHES_PER_PATIENT)
    else:
        paths_to_process = all_image_paths

    resize = transforms.Resize((IMG_SIZE, IMG_SIZE))
    to_tensor = transforms.ToTensor()
    
    positive_count = 0
    total_valid = 0

    for img_path in paths_to_process:
        try:
            with Image.open(img_path) as img:
                img = img.convert('RGB')
                img = resize(img)
                tensor = to_tensor(img).unsqueeze(0).to(device)
                
                with torch.no_grad(): outputs = model(tensor)
                if isinstance(outputs, tuple): reconstruction = outputs[0]
                else: reconstruction = outputs

                if MODEL_TYPE == 'AE':
                    h_in = rgb_to_hsv_h_channel_torch(tensor)
                    h_rec = rgb_to_hsv_h_channel_torch(reconstruction)
                    diff_sq = (h_in - h_rec) ** 2
                else:
                    diff_sq = (tensor - reconstruction) ** 2

                if ERROR_METRIC == 'MSE':
                    score = torch.mean(diff_sq).item()
                elif ERROR_METRIC == 'P99':
                    flat_diff = diff_sq.view(-1)
                    score = torch.quantile(flat_diff, 0.999).item()
                
                if score > MANUAL_PATCH_THRESHOLD:
                    positive_count += 1
                total_valid += 1
        except: continue

    if total_valid == 0: return None
    return positive_count / total_valid

def get_optimal_threshold(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    J = tpr - fpr
    ix = np.argmax(J)
    return thresholds[ix]

# ==========================================
# 3. PLOTTING ENHANCED
# ==========================================

def plot_confusion_matrix(y_true, y_pred, title, subtitle, filename):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Sa (0)', 'Malalt (1)'],
                yticklabels=['Sa (0)', 'Malalt (1)'])
    plt.xlabel('Predicció')
    plt.ylabel('Realitat')
    plt.title(title, fontsize=12, fontweight='bold')
    plt.suptitle(subtitle, fontsize=8, y=0.92)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()

def plot_roc_and_table_enhanced(tprs, aucs, mean_fpr, df_results, title, subtitle, filename, color='b', is_holdout=False):
    """
    Genera ROC Curve + Taula de Resultats amb Subtítol detallat.
    Mostra línies individuals per als folds de CV, però no per al bootstrap del Holdout.
    """
    fig, (ax_roc, ax_table) = plt.subplots(1, 2, figsize=(16, 10), gridspec_kw={'width_ratios': [3, 1]})
    
    # --- 1. ROC CURVE ---
    
    # A. PINTAR LÍNIES INDIVIDUALS (Només si és CV, no Holdout/Bootstrap)
    if not is_holdout:
        # Iterem sobre els tprs i els aucs per pintar cada fold
        for i, tpr_fold in enumerate(tprs):
            # Assegurem que tenim l'AUC corresponent (si la llista aucs té la mateixa mida)
            fold_auc = aucs[i] if i < len(aucs) else 0.0
            ax_roc.plot(mean_fpr, tpr_fold, lw=1, alpha=0.3, 
                        label=f'Fold {df_results.iloc[i]["ID"] if i < len(df_results) else i+1} (AUC = {fold_auc:.2f})')

    # B. CÀLCUL DE MITJANA I DESVIACIÓ ESTÀNDARD
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    
    # Desviació Estàndard exacta punt a punt
    std_tpr = np.std(tprs, axis=0)

    label_txt = r'Mean ROC (AUC = %0.3f $\pm$ %0.2f)' % (mean_auc, std_auc)
    
    # C. PINTAR MITJANA (Línia sòlida i més gruixuda)
    # Si volem suavitzar (SMOOTH_ROC és global, assegura't que és accessible o passa'l com argument)
    if 'SMOOTH_ROC' in globals() and globals()['SMOOTH_ROC']:
        x_smooth = np.linspace(0, 1, 300)
        try:
            pch = PchipInterpolator(mean_fpr, mean_tpr)
            y_smooth = np.clip(pch(x_smooth), 0, 1)
            ax_roc.plot(x_smooth, y_smooth, color=color, label=label_txt, lw=2.5)
        except:
            ax_roc.plot(mean_fpr, mean_tpr, color=color, label=label_txt, lw=2.5)
    else:
        ax_roc.plot(mean_fpr, mean_tpr, color=color, label=label_txt, lw=2.5)

    # D. PINTAR OMBRA (STD EXACTA)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax_roc.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.2, label=r'$\pm$ 1 std. dev.')

    # E. REFERÈNCIA I FORMAT
    ax_roc.plot([0, 1], [0, 1], linestyle='--', lw=1.5, color='black', alpha=.6)
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.set_title(title, fontsize=12, fontweight='bold')
    
    # Llegenda més petita perquè hi caben moltes línies
    ax_roc.legend(loc="lower right", fontsize='small')
    ax_roc.grid(alpha=0.3)

    # --- 2. TABLE ---
    ax_table.axis('off')
    
    col_name = "Set" if is_holdout else "Fold"
    cols = [col_name, "Acc", "Sens", "Spec"]
    cell_text = []
    
    for _, row in df_results.iterrows():
        row_id = str(row['ID'])
        cell_text.append([
            row_id, 
            f"{row['Acc']:.4f}", 
            f"{row['Sens']:.4f}", 
            f"{row['Spec']:.4f}"
        ])
        
    if not is_holdout and len(df_results) > 1:
        mean_row = df_results.mean(numeric_only=True)
        std_row = df_results.std(numeric_only=True)
        cell_text.append(["MEAN", f"{mean_row['Acc']:.4f}", f"{mean_row['Sens']:.4f}", f"{mean_row['Spec']:.4f}"])
        cell_text.append(["STD", f"±{std_row['Acc']:.4f}", f"±{std_row['Sens']:.4f}", f"±{std_row['Spec']:.4f}"])
    
    table = ax_table.table(cellText=cell_text, colLabels=cols, loc='center', cellLoc='center')
    table.scale(1, 1.5)
    table.auto_set_font_size(False); table.set_fontsize(10)
    
    if not is_holdout and len(df_results) > 1:
        for i in range(len(cols)):
            table[(len(cell_text)-2, i)].set_facecolor("#e6e6e6") # Mean
            table[(len(cell_text)-1, i)].set_facecolor("#e6e6e6") # Std

    # SUBTÍTOL GLOBAL
    plt.suptitle(subtitle, fontsize=10, y=0.95)
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    print(f"Gràfica guardada a: {os.path.join(OUTPUT_DIR, filename)}")
    plt.close()

# ==========================================
# 4. MAIN LOOP
# ==========================================

def main():
    print(f"--- INICIANT DIAGNOSI FINAL ---")
    
    # 1. Carregar Model
    config = AEConfigs(config_id='1', input_channels=3)
    if MODEL_TYPE == 'AE':
        model = AutoEncoderCNN(config.net_paramsEnc, config.inputmodule_paramsDec, config.net_paramsDec)
    elif MODEL_TYPE == 'VAE':
        try:
            model = VariationalAutoEncoderCNN(
                inputmodule_paramsEnc=config.inputmodule_paramsEnc, 
                net_paramsEnc=config.net_paramsEnc, 
                inputmodule_paramsDec=config.inputmodule_paramsDec, 
                net_paramsDec=config.net_paramsDec,
                latent_dim=LATENT_DIM, img_size=IMG_SIZE
            )
        except: sys.exit("Error VAE params")

    if not os.path.exists(MODEL_PATH): sys.exit(f"Model no trobat: {MODEL_PATH}")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE); model.eval()
    
    # 2. Dades
    df_diagnosis = pd.read_csv(CSV_DIAGNOSIS_PATH)
    df_diagnosis['Label'] = df_diagnosis['DENSITAT'].apply(lambda x: 0 if str(x).upper() == 'NEGATIVA' else 1)
    gt_dict = dict(zip(df_diagnosis['CODI'], df_diagnosis['Label']))
    
    with open(FOLDS_JSON_PATH, 'r') as f: folds_data = json.load(f)
        
    # --- CROSS-VALIDATION ---
    print(f"\n[1/2] Cross-Validation (Calculant Ratios)...")
    all_cv_y_true = []
    all_cv_y_ppp = []
    fold_data_storage = [] 

    for i, fold in enumerate(folds_data):
        test_patients = fold['test']
        print(f"  > Fold {i+1}/{len(folds_data)}...")
        y_true_fold = []
        y_ppp_fold = []
        for pat_id in tqdm(test_patients, leave=False):
            if pat_id not in gt_dict: continue
            ratio = calculate_patient_ppp(pat_id, CV_PATCHES_ROOT, model, DEVICE)
            if ratio is not None:
                y_true_fold.append(gt_dict[pat_id])
                y_ppp_fold.append(ratio)
        
        fold_data_storage.append({'y_true': y_true_fold, 'y_score': y_ppp_fold})
        all_cv_y_true.extend(y_true_fold)
        all_cv_y_ppp.extend(y_ppp_fold)

    print("\nCalculant el Percentatge Òptim (PPP Threshold)...")
    best_ppp_threshold = get_optimal_threshold(all_cv_y_true, all_cv_y_ppp)
    print(f"*** PPP THRESHOLD: {best_ppp_threshold:.4f} ***")
    
    # PREPARAR SUBTÍTOL
    plot_subtitle = (
        f"Model: {MODEL_TYPE} | Size: {IMG_SIZE}px | Metric: {ERROR_METRIC}\n"
        f"Patches/Pat: {MAX_PATCHES_PER_PATIENT} | Patch Thr: {MANUAL_PATCH_THRESHOLD} | PPP Thr: {best_ppp_threshold:.4f}"
    )

    # GRÀFIQUES CV
    results_list = []
    cv_tprs = []
    cv_aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    
    for i, data in enumerate(fold_data_storage):
        y_true = np.array(data['y_true'])
        y_score = np.array(data['y_score'])
        y_pred = (y_score >= best_ppp_threshold).astype(int)
        
        plot_confusion_matrix(y_true, y_pred, f"CM Fold {i+1}", plot_subtitle, f"cm_fold_{i+1}.png")

        if len(np.unique(y_true)) > 1:
            fpr, tpr, _ = roc_curve(y_true, y_score)
            cv_aucs.append(auc(fpr, tpr))
            cv_tprs.append(np.interp(mean_fpr, fpr, tpr))
            cv_tprs[-1][0] = 0.0
            
        acc = accuracy_score(y_true, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
        sens = tp/(tp+fn) if (tp+fn)>0 else 0
        spec = tn/(tn+fp) if (tn+fp)>0 else 0
        results_list.append({'ID': i+1, 'Acc': acc, 'Sens': sens, 'Spec': spec})

    df_res = pd.DataFrame(results_list)
    
    plot_roc_and_table_enhanced(cv_tprs, cv_aucs, mean_fpr, df_res, 
                                "Cross-Validation ROC", plot_subtitle, 
                                "cv_roc_final.png", color='darkorange', is_holdout=False)

    # --- HOLDOUT ---
    print(f"\n[2/2] HoldOut...")
    holdout_folders = glob.glob(os.path.join(HOLDOUT_PATCHES_ROOT, "*"))
    holdout_pat_ids = list(set([os.path.basename(p).split('_')[0] for p in holdout_folders]))
    
    ho_y_true = []
    ho_y_score = []
    
    for pat_id in tqdm(holdout_pat_ids):
        if pat_id not in gt_dict: continue
        ratio = calculate_patient_ppp(pat_id, HOLDOUT_PATCHES_ROOT, model, DEVICE)
        if ratio is not None:
            ho_y_true.append(gt_dict[pat_id])
            ho_y_score.append(ratio)
            
    ho_y_true = np.array(ho_y_true)
    ho_y_score = np.array(ho_y_score)
    
    if len(ho_y_true) > 0:
        ho_y_pred = (ho_y_score >= best_ppp_threshold).astype(int)
        
        # 1. Confusion Matrix
        plot_confusion_matrix(ho_y_true, ho_y_pred, "HoldOut CM", plot_subtitle, "holdout_cm.png")
        
        # 2. Calcular mètriques Holdout
        acc = accuracy_score(ho_y_true, ho_y_pred)
        tn, fp, fn, tp = confusion_matrix(ho_y_true, ho_y_pred, labels=[0,1]).ravel()
        sens = tp/(tp+fn) if (tp+fn)>0 else 0
        spec = tn/(tn+fp) if (tn+fp)>0 else 0
        
        # DataFrame per a la taula
        df_ho_res = pd.DataFrame([{'ID': 'Holdout', 'Acc': acc, 'Sens': sens, 'Spec': spec}])

        # 3. Bootstrapping per ROC (per tenir l'ombrejat Std Dev)
        ho_tprs = []
        ho_aucs = []
        for _ in range(1000):
            indices = resample(np.arange(len(ho_y_true)), replace=True, n_samples=len(ho_y_true))
            if len(np.unique(ho_y_true[indices])) < 2: continue 
            fpr, tpr, _ = roc_curve(ho_y_true[indices], ho_y_score[indices])
            ho_aucs.append(auc(fpr, tpr))
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            ho_tprs.append(interp_tpr)
        
        # 4. Plot ROC Holdout amb Taula
        plot_roc_and_table_enhanced(ho_tprs, ho_aucs, mean_fpr, df_ho_res, 
                                    "HoldOut ROC", plot_subtitle, 
                                    "holdout_roc_final.png", color='green', is_holdout=True)
        
        print(f"\nResultats HoldOut:")
        print(f"Acc: {acc:.4f} | Sens: {sens:.4f} | Spec: {spec:.4f}")
    else:
        print("No Holdout Data.")

    print(f"\nFet. Gràfiques a: {OUTPUT_DIR}")

if __name__ == '__main__':
    main()