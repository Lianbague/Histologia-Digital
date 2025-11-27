# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import sys
import pandas as pd
from sklearn.metrics import roc_curve, auc, f1_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import numpy as np
import time
import re
import colorsys
import random
from tqdm import tqdm
import warnings
import json

# Silenciar warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# --- IMPORTS DEL MODEL ---
from _1_AE_train.ae_models import AutoEncoderCNN, AEConfigs, VariationalAutoEncoderCNN

# ==========================================
# 1. FUNCIONS AUXILIARS (GPU / DATASET)
# ==========================================

def rgb_to_hsv_h_channel_torch(image_tensor):
    """ Calcula només el canal H (Hue) utilitzant tensors a GPU. """
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

class InMemoryEvalDataset(torch.utils.data.Dataset):
    def __init__(self, df, base_dir, img_size=256):
        self.data = []
        self.labels = []
        self.indices = [] 
        
        print(f"Pre-carregant {len(df)} imatges a RAM ({img_size}x{img_size})...")
        
        resize = transforms.Resize((img_size, img_size))
        to_tensor = transforms.ToTensor()
        
        success_count = 0
        
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            try:
                pat_section = row['Pat_Section']
                window_id = row['Window_ID']
                # Regex per gestionar noms d'arxius
                match = re.match(r"(\d+)(_Aug\d*)?$", str(window_id))
                if not match: continue
                
                num_part = int(match.group(1))
                aug_part = match.group(2) if match.group(2) else ""
                filename = f"{num_part:05d}{aug_part}.png"
                
                path = os.path.join(base_dir, pat_section, filename)
                
                if not os.path.exists(path):
                    continue
                
                with Image.open(path) as img:
                    img = img.convert('RGB')
                    img = resize(img)
                    # Guardem com uint8 per estalviar RAM, convertim a float a __getitem__
                    tensor = (to_tensor(img) * 255).to(torch.uint8)
                    
                    self.data.append(tensor)
                    self.labels.append(int(row['Presence'])) 
                    self.indices.append(idx)
                    success_count += 1
                    
            except Exception as e:
                continue

        print(f"Carrèga finalitzada. {success_count} imatges llestes.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_uint8 = self.data[idx]
        img_float = img_uint8.float() / 255.0
        return img_float, self.labels[idx], self.indices[idx]

def find_optimal_cutoff(true_labels, scores):
    fpr, tpr, thresholds = roc_curve(true_labels, scores)
    best_f1 = 0
    opt_tau = 0
    opt_fpr = 0
    opt_tpr = 0
    
    for i, t in enumerate(thresholds):
        y_pred = (scores >= t).astype(int)
        f1 = f1_score(true_labels, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            opt_tau = t
            opt_fpr = fpr[i]
            opt_tpr = tpr[i]
    return opt_tau, opt_fpr, opt_tpr, best_f1

def get_stratified_folds(df_annotated, num_folds, save_path):
    patient_groups = df_annotated.groupby('PatID')['Presence'].max()
    unique_patients = patient_groups.index.values
    unique_labels = patient_groups.values
    
    num_pos = np.sum(unique_labels == 1)
    
    if os.path.exists(save_path):
        print(f"Cerrgant folds existents de: {save_path}")
        with open(save_path, 'r') as f:
            folds_data = json.load(f)
        return folds_data, unique_patients

    print(f"Generant nous folds estratificats...")
    actual_splits = num_pos if num_pos < num_folds else num_folds
        
    skf = StratifiedKFold(n_splits=actual_splits, shuffle=True, random_state=42)
    folds_data = []
    
    for train_idx, test_idx in skf.split(unique_patients, unique_labels):
        folds_data.append({
            'train': unique_patients[train_idx].tolist(),
            'test': unique_patients[test_idx].tolist()
        })
        
    with open(save_path, 'w') as f:
        json.dump(folds_data, f)
    return folds_data, unique_patients

def plot_reconstruction(input_tensor, reconstruction, error_score, save_dir, filename_prefix, pat_label, metric_name):
    os.makedirs(save_dir, exist_ok=True)
    
    inp_img = input_tensor.permute(1, 2, 0).cpu().numpy()
    recon_img = reconstruction.permute(1, 2, 0).cpu().numpy()
    recon_img = np.clip(recon_img, 0, 1) # Clip per seguretat
    
    h_in = rgb_to_hsv_h_channel_torch(input_tensor.unsqueeze(0)).squeeze(0)
    h_rec = rgb_to_hsv_h_channel_torch(reconstruction.unsqueeze(0)).squeeze(0)
    
    # Per visualitzar sempre mostrem el mapa d'error absolut
    error_map = torch.abs(h_rec - h_in).cpu().numpy()
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(inp_img)
    plt.title(f"Original (Label: {pat_label})")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(recon_img)
    plt.title("Reconstrucció")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(error_map, cmap='hot')
    plt.colorbar(label='Abs Error H')
    plt.title(f"Mapa Error (Score {metric_name}: {error_score:.6f})")
    plt.axis('off')
    
    plt.suptitle(f"Anomaly Score ({metric_name}): {error_score:.6f}")
    
    save_path = os.path.join(save_dir, f"{filename_prefix}_score_{error_score:.6f}.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


# ==========================================
# 2. MAIN
# ==========================================

def main():
    # ---------------------------------------------------------
    # --- CONFIGURACIÓ D'USUARI ---
    # ---------------------------------------------------------
    IMG_SIZE = 256            # Opcions: 128 o 256
    MODEL_TYPE = 'VAE'        # Opcions: 'AE' o 'VAE'
    ERROR_METRIC = 'MSE'      # Opcions: 'MSE' (Mean Squared Error) o 'P99' (Percentil 99.99)
    
    # Rutes Base (Ajustar)
    BASE_EXPORT = '/export/fhome/maed03'
    CSV_ANNOTATED_PATH = '/export/fhome/maed/HelicoDataSet/HP_WSI-CoordAllAnnotatedPatches.xlsx'
    BASE_IMAGE_DIR = '/export/fhome/maed/HelicoDataSet/CrossValidation/Annotated'
    FOLDS_JSON_PATH = os.path.join(BASE_EXPORT, '_2_fix_threshold/folds_distribution_10folds.json')
    
    # Configuración Automática de Rutas basada en opciones
    if MODEL_TYPE == 'AE':
        # Rutes típiques para AE
        if IMG_SIZE == 256:
            MODEL_PATH = os.path.join(BASE_EXPORT, '_1_AE_train/ae_L1Loss_256.pth')
        else:
            MODEL_PATH = os.path.join(BASE_EXPORT, '_1_AE_train/ae_L1Loss_128.pth')
    else:
        # Rutes típiques para VAE
        if IMG_SIZE == 256:
            MODEL_PATH = os.path.join(BASE_EXPORT, '_1_AE_train/vae_256.pth')
        else:
            MODEL_PATH = os.path.join(BASE_EXPORT, '_1_AE_train/vae_128.pth')

    # Noms de sortida
    suffix = f"{MODEL_TYPE}_{IMG_SIZE}_{ERROR_METRIC}"
    ROC_CURVE_SAVE_PATH = os.path.join(BASE_EXPORT, f'_2_fix_threshold/roc_{suffix}.png')
    VISUAL_SAVE_DIR = os.path.join(BASE_EXPORT, f'_2_fix_threshold/recons_{suffix}')
    
    # Paràmetres Model
    NUM_FOLDS = 10
    # Ajustem Batch Size si es VAE + 256 per evitar OOM, sino 128
    BATCH_SIZE = 32 if (MODEL_TYPE == 'VAE' and IMG_SIZE == 256) else 128
    LATENT_DIM = 128
    
    print(f"\n=== CONFIGURACIÓ ACTIVA ===")
    print(f"Model: {MODEL_TYPE}")
    print(f"Imatge: {IMG_SIZE}x{IMG_SIZE}")
    print(f"Mètrica: {ERROR_METRIC}")
    print(f"Carregant model de: {MODEL_PATH}")
    print(f"Guardant resultats a: {VISUAL_SAVE_DIR}")
    print("============================\n")

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Dispositiu: {DEVICE}")

    # --- A. Carrega de Dades ---
    try:
        df_annotated = pd.read_excel(CSV_ANNOTATED_PATH)
    except:
        df_annotated = pd.read_csv(CSV_ANNOTATED_PATH, encoding='latin-1')

    # Neteja de columnes
    df_annotated.columns = [col.strip().replace(' ', '_').replace('Pat_ID', 'Pat_ID') for col in df_annotated.columns]
    df_annotated['Pat_Section'] = df_annotated['Pat_ID'].astype(str).str.strip() + '_' + df_annotated['Section_ID'].astype(str).str.strip()
    df_annotated['PatID'] = df_annotated['Pat_ID'].astype(str).str.strip()
    df_annotated['Presence'] = df_annotated['Presence'].apply(lambda x: 1 if x == 1 else 0)

    # Dataset
    full_dataset = InMemoryEvalDataset(df_annotated, BASE_IMAGE_DIR, img_size=IMG_SIZE)
    loaded_indices = full_dataset.indices
    patient_ids_per_sample = np.array([df_annotated.iloc[i]['PatID'] for i in loaded_indices])
    
    # Folds
    folds_data, unique_patients = get_stratified_folds(df_annotated, NUM_FOLDS, FOLDS_JSON_PATH)

    # --- B. Inicialització del Model ---
    config = AEConfigs(config_id='1', input_channels=3)
    
    if MODEL_TYPE == 'AE':
        model = AutoEncoderCNN(config.net_paramsEnc, config.inputmodule_paramsDec, config.net_paramsDec)
    elif MODEL_TYPE == 'VAE':
        # VAE requereix img_size y latent_dim explícits en aquesta implementació
        try:
            model = VariationalAutoEncoderCNN(
                inputmodule_paramsEnc=config.inputmodule_paramsEnc, 
                net_paramsEnc=config.net_paramsEnc, 
                inputmodule_paramsDec=config.inputmodule_paramsDec, 
                net_paramsDec=config.net_paramsDec,
                latent_dim=LATENT_DIM,
                img_size=IMG_SIZE
            )
        except TypeError:
            print("Error al instanciar VAE. Verifica ae_models.py.")
            sys.exit(1)
            
    if not os.path.exists(MODEL_PATH):
        sys.exit(f"ERROR: No es troba el model en {MODEL_PATH}")
        
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    # --- C. Loop d'Avaluació ---
    mean_fpr = np.linspace(0, 1, 100)
    all_scores_global = []
    all_labels_global = []
    
    tprs_patch_list = []
    fold_accuracies = []
    fold_aucs = []
    
    os.makedirs(VISUAL_SAVE_DIR, exist_ok=True)

    for i_fold, fold_info in enumerate(folds_data):
        test_pat_ids = fold_info['test']
        mask_test = np.isin(patient_ids_per_sample, test_pat_ids)
        test_indices = np.where(mask_test)[0]
        
        if len(test_indices) == 0: continue

        test_loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(full_dataset, test_indices), 
            batch_size=BATCH_SIZE, shuffle=False
        )
        
        fold_scores = []
        fold_labels = []
        
        pos_samples_buffer = []
        neg_samples_buffer = []
        
        with torch.no_grad():
            for inputs, labels, indices in tqdm(test_loader, desc=f"Fold {i_fold+1}", leave=False):
                inputs = inputs.to(DEVICE)
                
                # Forward pass
                outputs = model(inputs)
                # Gestionar sortida VAE (tupla) vs AE (tensor)
                if isinstance(outputs, tuple):
                    reconstructions = outputs[0]
                else:
                    reconstructions = outputs
                
                # --- CÁLCUL D'ERROR ---
                h_in = rgb_to_hsv_h_channel_torch(inputs)
                h_rec = rgb_to_hsv_h_channel_torch(reconstructions)
                
                
#                # Netejar NaNs si apareixen (més comú a VAE)
#                if torch.isnan(diff_sq).any():
#                    diff_sq = torch.nan_to_num(diff_sq, nan=1.0)
                
                if ERROR_METRIC == 'MSE':
                    if MODEL_TYPE == 'AE':
                        diff_sq = (h_in - h_rec) ** 2
                    else:
                        diff_sq = (inputs - reconstructions) ** 2
                    # Mitjana del error al quadrat per imatge
                    batch_scores = torch.mean(diff_sq, dim=[1, 2])
                    
                elif ERROR_METRIC == 'P99':
                    if MODEL_TYPE == 'AE':
                        diff_sq = (h_in - h_rec) ** 2
                    else:
                        diff_sq = (inputs - reconstructions) ** 2
                    # Percentil 99.99
                    flat_diff = diff_sq.view(inputs.size(0), -1)
                    batch_scores = torch.quantile(flat_diff, 0.9999, dim=1)
                
                mse_cpu = batch_scores.cpu().numpy()
                lbl_cpu = labels.cpu().numpy()
                inp_cpu = inputs.cpu()
                rec_cpu = reconstructions.cpu()
                
                fold_scores.extend(mse_cpu.tolist())
                fold_labels.extend(lbl_cpu.tolist())
                
                # Guardar mostres pel plot
                for k in range(len(lbl_cpu)):
                    item = (inp_cpu[k], rec_cpu[k], batch_scores[k].item(), lbl_cpu[k])
                    if lbl_cpu[k] == 1 and len(pos_samples_buffer) < 3:
                        pos_samples_buffer.append(item)
                    elif lbl_cpu[k] == 0 and len(neg_samples_buffer) < 3:
                        neg_samples_buffer.append(item)

        # Plot mostres del fold
        samples_to_plot = pos_samples_buffer + neg_samples_buffer
        for idx_img, (inp, rec, err, lbl) in enumerate(samples_to_plot):
            lbl_str = "POS" if lbl==1 else "NEG"
            prefix = f"Fold{i_fold+1}_Sample{idx_img}_{lbl_str}"
            plot_reconstruction(inp, rec, err, VISUAL_SAVE_DIR, prefix, lbl, ERROR_METRIC)

        # Afegir a globals
        all_scores_global.extend(fold_scores)
        all_labels_global.extend(fold_labels)
        
        # Estadístiques del fold
        f_scores = np.array(fold_scores)
        f_labels = np.array(fold_labels)
        
        # Noetejar posibles NaNs a numpy abans de metriques
        valid_idx = ~np.isnan(f_scores)
        f_scores = f_scores[valid_idx]
        f_labels = f_labels[valid_idx]
        
        if len(np.unique(f_labels)) > 1:
            fpr_f, tpr_f, _ = roc_curve(f_labels, f_scores)
            fold_auc = auc(fpr_f, tpr_f)
            fold_aucs.append(fold_auc)
            
            interp_tpr = np.interp(mean_fpr, fpr_f, tpr_f)
            interp_tpr[0] = 0.0
            tprs_patch_list.append(interp_tpr)
            
            t_tau, _, _, _ = find_optimal_cutoff(f_labels, f_scores)
            fold_preds = (f_scores >= t_tau).astype(int)
            fold_accuracies.append(accuracy_score(f_labels, fold_preds))

    # --- D. Gràfica Final y Taula ---
    print(f"\n--- Generant Gràfica Final ({suffix}) ---")
    fig, (ax_roc, ax_table) = plt.subplots(1, 2, figsize=(16, 10), gridspec_kw={'width_ratios': [3, 1]})
    
    scores_arr = np.array(all_scores_global)
    labels_arr = np.array(all_labels_global)
    
    # Neteja final
    valid_gl = ~np.isnan(scores_arr)
    scores_arr = scores_arr[valid_gl]
    labels_arr = labels_arr[valid_gl]
    
    # 1. ROC
    if tprs_patch_list:
        # A. Pintar les corbes individuals de cada FOLD
        for i, tpr_fold in enumerate(tprs_patch_list):
            roc_auc_fold = fold_aucs[i]
            ax_roc.plot(mean_fpr, tpr_fold, lw=1, alpha=0.3,
                        label=f'ROC Fold {i+1} (AUC = {roc_auc_fold:.2f})')

        # B. Calcular Mitjana i Std
        mean_tpr = np.mean(tprs_patch_list, axis=0)
        mean_tpr[-1] = 1.0
        
        # Càlcul de la Desviació Estàndard exacta
        std_tpr = np.std(tprs_patch_list, axis=0)
        
        mean_auc_val = np.mean(fold_aucs)
        std_auc_val = np.std(fold_aucs)
        
        # C. Pintar la corba MITJANA (més gruixuda i color sòlid)
        label_curve = r'Mean ROC (AUC = %0.3f $\pm$ %0.3f)' % (mean_auc_val, std_auc_val)
        ax_roc.plot(mean_fpr, mean_tpr, color='blue', lw=2, label=label_curve)
        
        # D. Pintar l'ombra (STD exacta: Mean + Std i Mean - Std)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax_roc.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.2,
                            label=r'$\pm$ 1 std. dev.')

    # Pintar el punt de tall òptim global
    opt_tau, opt_fpr, opt_tpr, _ = find_optimal_cutoff(labels_arr, scores_arr)
    ax_roc.plot(opt_fpr, opt_tpr, 'r*', markersize=15, label=f'Global Opt Tau: {opt_tau:.4f}')
    
    ax_roc.plot([0, 1], [0, 1], 'k--', lw=1)
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.set_title(f'ROC: {MODEL_TYPE} {IMG_SIZE}px - Metric: {ERROR_METRIC}')
    
    # Llegenda fora o ajustada (com hi ha moltes línies, fem la font petita)
    ax_roc.legend(loc="lower right", fontsize='small')
    ax_roc.grid(alpha=0.4)
    
    # 2. TAULA
    ax_table.axis('off')
    table_data = [[f"Fold {i+1}", f"{acc:.4f}"] for i, acc in enumerate(fold_accuracies)]
    table_data.append(["MEAN", f"{np.mean(fold_accuracies):.4f}"])
    table_data.append(["STD", f"± {np.std(fold_accuracies):.4f}"])
    
    table = ax_table.table(cellText=table_data, colLabels=["Fold", "Accuracy"], loc='center', cellLoc='center')
    table.scale(1, 2)
    table.auto_set_font_size(False); table.set_fontsize(10)
    
    # Pintar files de resum
    for i in range(2):
        table[(len(table_data)-1, i)].set_facecolor("#e6e6e6")
        table[(len(table_data), i)].set_facecolor("#e6e6e6")

    plt.tight_layout()
    plt.savefig(ROC_CURVE_SAVE_PATH)
    print(f"Gràfica guardada a: {ROC_CURVE_SAVE_PATH}")

if __name__ == '__main__':
    main()