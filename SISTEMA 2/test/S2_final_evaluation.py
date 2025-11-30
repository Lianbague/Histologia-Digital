import os
import glob
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, accuracy_score, recall_score, auc
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# Importar modelos y dataset
from S2_MIL.S2_models import NeuralNetwork_withAttention
from S2_HoldOut_test.S2_HoldOut_dataset import HoldOutDataset

# ==========================================
#   CONFIGURACIÓN GLOBAL
# ==========================================

# ELIGE EL MODELO: 'ResNet', 'Triplet' o 'AE'
MODEL_TYPE = 'AE' 

FIXED_THRESHOLD = 0.5 

# RUTAS BASE
BASE_DIR = '/export/fhome/maed03'
CSV_PATH = '/export/fhome/maed/HelicoDataSet/PatientDiagnosis.csv'
OUTPUT_DIR = os.path.join(BASE_DIR,"S2_HoldOut_test", "S2_HoldOut_final_evaluation_T0.5") # Modificado para no sobrescribir
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Rutas según el modelo
if MODEL_TYPE == 'ResNet':
    FEATURE_DIR = os.path.join(BASE_DIR, "Features_HoldOut_ResNet")
    MODELS_AND_THRESHOLDS = [
        ["S2_best_model_ResNet_fold1.pth",0.5],["S2_best_model_ResNet_fold2.pth",0.5],
        ["S2_best_model_ResNet_fold3.pth",0.5],["S2_best_model_ResNet_fold4.pth",0.5],
        ["S2_best_model_ResNet_fold5.pth",0.5], ["S2_best_model_ResNet_fold6.pth",0.5],
        ["S2_best_model_ResNet_fold7.pth",0.5], ["S2_best_model_ResNet_fold8.pth",0.5],
        ["S2_best_model_ResNet_fold9.pth",0.5], ["S2_best_model_ResNet_fold10.pth",0.5],]

elif MODEL_TYPE == 'Triplet':
    FEATURE_DIR = os.path.join(BASE_DIR, "Features_HoldOut_Triplet128")
    MODELS_AND_THRESHOLDS = [
        ["S2_best_model_Triplet_fold1.pth", 0.5],["S2_best_model_Triplet_fold2.pth", 0.5],
        ["S2_best_model_Triplet_fold3.pth", 0.5],["S2_best_model_Triplet_fold4.pth", 0.5],
        ["S2_best_model_Triplet_fold5.pth", 0.5], ["S2_best_model_Triplet_fold6.pth", 0.5],
        ["S2_best_model_Triplet_fold7.pth", 0.5], ["S2_best_model_Triplet_fold8.pth", 0.5],
        ["S2_best_model_Triplet_fold9.pth", 0.5], ["S2_best_model_Triplet_fold10.pth", 0.5],]

elif MODEL_TYPE == 'AE':
    FEATURE_DIR = os.path.join(BASE_DIR, "Features_HoldOut_AE")
    MODELS_AND_THRESHOLDS = [
        ["S2_best_model_AE_fold1.pth",0.5],["S2_best_model_AE_fold2.pth",0.5],
        ["S2_best_model_AE_fold3.pth",0.5],["S2_best_model_AE_fold4.pth",0.5],
        ["S2_best_model_AE_fold5.pth",0.5], ["S2_best_model_AE_fold6.pth",0.5],
        ["S2_best_model_AE_fold7.pth",0.5], ["S2_best_model_AE_fold8.pth",0.5],
        ["S2_best_model_AE_fold9.pth",0.5], ["S2_best_model_AE_fold10.pth",0.5],]
else:
    raise ValueError(f"MODEL_TYPE '{MODEL_TYPE}' no reconocido.")

# Nombres de archivos de salida
OUTPUT_PRED_CSV = os.path.join(OUTPUT_DIR, f"holdout_predictions_{MODEL_TYPE}_T0.5.csv") 

print(f"--- CONFIGURACION ---")
print(f"Modelo: {MODEL_TYPE}")
print(f"Features: {FEATURE_DIR}")
print(f"Umbral Fijo: {FIXED_THRESHOLD}")
print(f"Salida CSV: {OUTPUT_PRED_CSV}")
print(f"---------------------")


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
    print(f"Matriz Confusión guardada: {filename}")

def load_model(filename, input_dim):
    path_options = [
        os.path.join(BASE_DIR, "S2_MIL", filename),
        os.path.join(BASE_DIR, filename)
    ]
    
    final_path = None
    for p in path_options:
        if os.path.exists(p):
            final_path = p
            break
    
    if not final_path:
        print(f"Modelo no encontrado: {filename}")
        return None

    model = NeuralNetwork_withAttention(
        input_dim=input_dim,
        project_dim=512,
        decom_space=128,
        attention_branches=1,
        attention_type="GatedAttention"
    ).to(DEVICE)

    try:
        model.load_state_dict(torch.load(final_path, map_location=DEVICE))
        model.eval()
        return model
    except Exception as e:
        print(f"rror cargando pesos de {filename}: {e}")
        return None

def generate_individual_roc_with_thresholds(true_labels, fold_probs_list, thresholds_data, model_type):
    global FIXED_THRESHOLD # Usamos el umbral fijo
    
    print("\nGenerando ROC Curves Individuales con Umbral Fijo de 0.5 Aplicado...")
    roc_dir = os.path.join(OUTPUT_DIR, "Individual_ROC_Plots_T0.5")
    os.makedirs(roc_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    
    # Colores para distinguir los folds
    colors = plt.cm.get_cmap('hsv', len(fold_probs_list)) 
    
    # Iterar por cada fold
    for i, fold_probs in enumerate(fold_probs_list):
        fold_num = i + 1
        model_filename, _ = thresholds_data[i] # El umbral de thresholds_data ya no se usa aquí
        
        # 1. Calcular ROC
        fpr, tpr, thresholds_roc = roc_curve(true_labels, fold_probs)
        roc_auc = auc(fpr, tpr)
        
        # 2. Encontrar el punto más cercano al threshold de 0.5
        threshold_diffs = np.abs(thresholds_roc - FIXED_THRESHOLD)
        optimal_idx = np.argmin(threshold_diffs)

        # 3. Plotear la curva ROC del fold
        plt.plot(fpr, tpr, color=colors(i), lw=1.5, alpha=0.6, 
                 label=f'Fold {fold_num} (AUC = {roc_auc:.3f})')
        
        # 4. Marcar el Umbral Fijo de 0.5 Aplicado
        # Usamos un marcador diferente y solo etiquetamos una vez el 0.5
        marker_label = f'Umbral Fijo: {FIXED_THRESHOLD:.1f}' if i == 0 else '_nolegend_'
        plt.plot(fpr[optimal_idx], tpr[optimal_idx], 's', # Cuadrado para 0.5
                 color='black', markersize=8, 
                 label=marker_label, markeredgecolor='black', markeredgewidth=1.0)
        
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'HoldOut ROC Curves with Fixed Threshold (0.5) - {model_type}', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=8)
    plt.grid(True, alpha=0.3)
    
    filename = f'holdout_roc_individual_fixed_T0.5_{model_type}.png'
    plt.savefig(os.path.join(roc_dir, filename), dpi=300)
    plt.close()
    print(f"Gráfica ROC individual guardada: {filename}")


def run_individual_evaluation():
    print(f"--- INICIANDO EVALUACIÓN INDIVIDUAL (10 MODELS) - {MODEL_TYPE} con UMBRAL FIJO {FIXED_THRESHOLD} ---")
    
    if not os.path.exists(FEATURE_DIR):
        print(f"Error: No existe el directorio de features: {FEATURE_DIR}")
        sys.exit(1)

    df = pd.read_csv(CSV_PATH)
    ds = HoldOutDataset(FEATURE_DIR, df)
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    if len(ds) == 0: 
        print("Error: Dataset vacio.")
        return

    # 1. Obtener etiquetas y dimensión de entrada
    try:
        _, sample_feat, _ = ds[0]
        INPUT_DIM = sample_feat.shape[1]
        print(f"Dimensión detectada: {INPUT_DIM}")
    except Exception as e:
        print(f"Error accediendo al dataset: {e}")
        return

    true_labels = []
    
    # Primera pasada para obtener labels (solo se necesita una vez)
    # Usamos try-except dentro del bucle para saltar corruptos al vuelo
    print("Leyendo etiquetas reales...")
    for pid, features, label in loader:
        try:
            true_labels.append(label.item())
        except: continue
            
    true_labels = np.array(true_labels)

    all_results = []
    all_fold_probs = []

    # 2. Iterar sobre los 10 modelos
    for i, (model_filename, _) in enumerate(MODELS_AND_THRESHOLDS): # El umbral optimizado ya no se usa aquí
        fold_num = i + 1
        print(f"\n--- Evaluando Modelo {fold_num}: {model_filename} (T FIJO={FIXED_THRESHOLD:.3f}) ---")
        
        model = load_model(model_filename, INPUT_DIM)
        if model is None:
            continue

        # 3. Obtener probabilidades
        fold_probs = []
        with torch.no_grad():
            for pid, features, label in loader:
                try:
                    features = features.to(DEVICE)
                    prob, _ = model(features)
                    fold_probs.append(prob.item())
                except Exception as e:
                    print(f"Error inferencia {pid}: {e}")
                    # Si falla inferencia, ponemos 0.5 o saltamos, pero hay que mantener sincronia con labels
                    # Lo ideal seria limpiar el dataset antes
                    continue
        
        fold_probs = np.array(fold_probs)
        
        # Asegurar consistencia
        if len(fold_probs) != len(true_labels):
            print(f"⚠️ Warning: Longitud de probs ({len(fold_probs)}) != labels ({len(true_labels)})")
            # Recortar al mínimo común por si acaso
            min_len = min(len(fold_probs), len(true_labels))
            fold_probs = fold_probs[:min_len]
            current_labels = true_labels[:min_len]
        else:
            current_labels = true_labels

        # Guardamos para la gráfica final
        all_fold_probs.append(fold_probs)
        
        preds = (fold_probs >= FIXED_THRESHOLD).astype(int)
        
        # 5. Calcular métricas para el resumen
        acc = accuracy_score(current_labels, preds)
        try:
            auc_score = roc_auc_score(current_labels, fold_probs)
        except: auc_score = 0.5 # Si solo hay una clase
            
        tn, fp, fn, tp = confusion_matrix(current_labels, preds, labels=[0,1]).ravel()
        sens = tp/(tp+fn) if (tp+fn)>0 else 0
        spec = tn/(tn+fp) if (tn+fp)>0 else 0
        
        print(f"  AUC: {auc_score:.4f}, Acc: {acc:.4f}, Sens: {sens:.4f}, Spec: {spec:.4f}")

        all_results.append({
            "Fold": fold_num,
            "Model_File": model_filename,
            "Threshold_Used": FIXED_THRESHOLD, 
            "AUC": auc_score,
            "Accuracy": acc,
            "Sensitivity": sens,
            "Specificity": spec,
        })
        
        # 6. Generar Matriz de Confusión
        plot_confusion_matrix(
            current_labels, preds,
            title=f"Matriu de Confusió - Fold {fold_num} ({MODEL_TYPE})",
            subtitle=f"Model: {model_filename} | Umbral Fijo: {FIXED_THRESHOLD:.1f}\nSet: HoldOut ({len(current_labels)} pacients)",
            filename=f"cm_holdout_{MODEL_TYPE}_fold{fold_num}_T0.5.png" # Nombre de archivo ajustado
        )
        
    # 7. Guardar Resumen de Resultados
    if all_results:
        df_resumen = pd.DataFrame(all_results)
        df_resumen.to_csv(OUTPUT_PRED_CSV, index=False)
        print(f"\nResumen de resultados guardado en: {OUTPUT_PRED_CSV}")
        print("\nResumen por Fold:")
        print(df_resumen)

        # Generar ROC conjunta
        generate_individual_roc_with_thresholds(
            true_labels, 
            all_fold_probs, 
            MODELS_AND_THRESHOLDS, 
            MODEL_TYPE
        )

if __name__ == "__main__":
    run_individual_evaluation()