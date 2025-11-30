import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, accuracy_score, recall_score, auc
from scipy.interpolate import PchipInterpolator
import json
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import glob

# Importar tus modelos
from S2_MIL.S2_models import NeuralNetwork_withAttention 

# ==========================================
#   CONFIGURACIÓN GLOBAL
# ==========================================

# ELIGE EL MODELO: 'ResNet', 'Triplet' o 'AE'
MODEL_TYPE = 'AE' 
FIXED_THRESHOLD = 0.5 

# RUTAS BASE
BASE_DIR = '/export/fhome/maed03'
CSV_PATH = '/export/fhome/maed/HelicoDataSet/PatientDiagnosis.csv'
FOLDS_JSON_PATH = '/fhome/maed03/_2_fix_threshold/folds_distribution_10folds.json'

# Output Directorio (Cambio el nombre para que no sobrescriba HoldOut)
OUTPUT_DIR = os.path.join(BASE_DIR, "S2_MIL", f"Evaluation_CV_{MODEL_TYPE}_T0.5") 
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Determinación de rutas de Features y Modelos para CV
if MODEL_TYPE == 'ResNet':
    FEATURE_DIR = '/export/fhome/maed03/Features_ResNet'
    INPUT_DIM = 2048 # ResNet features son 2048
elif MODEL_TYPE == 'Triplet':
    FEATURE_DIR = '/export/fhome/maed03/Features_Triplet128'
    INPUT_DIM = 128 # Asumo la dimensión final del Triplet (ajusta si es 128)
elif MODEL_TYPE == 'AE':
    FEATURE_DIR = '/export/fhome/maed03/Features_AE'
    INPUT_DIM = 32768
else:
    raise ValueError(f"MODEL_TYPE '{MODEL_TYPE}' no reconocido.")

# Nombres de archivos de modelos
MODELS_FILES = [f"S2_best_model_{MODEL_TYPE}_fold{i}.pth" for i in range(1, 11)]

print(f"--- CONFIGURACION ---")
print(f"Modelo: {MODEL_TYPE}")
print(f"Features CV: {FEATURE_DIR}")
print(f"Dimensión de Entrada: {INPUT_DIM}")
print(f"Umbral Fijo: {FIXED_THRESHOLD}")
print(f"Salida: {OUTPUT_DIR}")
print(f"---------------------")


# ==========================================
#   CLASES Y FUNCIONES DE CARGA DE DATOS
# (Adaptadas de S2_train_mil_kfold.py)
# ==========================================

class PatientFeatureDataset(Dataset):
    def __init__(self, patient_ids, features_dir, df_diagnosis):
        self.patient_ids = patient_ids
        self.features_dir = features_dir
        self.df = df_diagnosis
    
    def __len__(self): return len(self.patient_ids)
    
    def __getitem__(self, idx):
        pat_id = self.patient_ids[idx]
        
        # Carregar el tensor de caracteristiques (bag)
        features_path = os.path.join(self.features_dir, f"{pat_id}.pt")
        try:
            features = torch.load(features_path, weights_only=True)
        except Exception as e:
            # Manejo de error si el archivo no existe o está corrupto
            print(f"Error cargando features para {pat_id}: {e}")
            # Devolver tensores placeholder para evitar que el DataLoader falle
            # NOTA: Esto hará que el resultado de ese paciente sea incorrecto.
            features = torch.randn(1, INPUT_DIM) 

        # Obtenir etiqueta (Bacteria = 1, Negativa = 0)
        label_str = self.df[self.df['CODI'] == pat_id]['DENSITAT'].values[0]
        label = 1.0 if label_str != 'NEGATIVA' else 0.0
        
        return pat_id, features, torch.tensor([label], dtype=torch.float32)

def load_folds_patients(df_diagnosis, features_dir):
    if not os.path.exists(FOLDS_JSON_PATH):
        raise FileNotFoundError(f"ERROR: Archivo de folds no encontrado en {FOLDS_JSON_PATH}")
    
    with open(FOLDS_JSON_PATH, 'r') as f:
        folds_data = json.load(f)

    available_pats = {f.replace('.pt', '') for f in os.listdir(features_dir) if f.endswith('.pt')}
    
    cv_splits = {}
    for fold_idx, fold_info in enumerate(folds_data):
        fold_num = fold_idx + 1
        
        # Solo necesitamos los pacientes de validación (test)
        val_pats_json = fold_info['test']
        
        # Filtrar para asegurar que tenemos features y etiqueta para el paciente
        valid_val_pats = [p for p in val_pats_json if p in available_pats and p in df_diagnosis['CODI'].values]
        
        if not valid_val_pats:
            print(f"Advertencia: Fold {fold_num} tiene 0 pacientes válidos para validación.")
            continue
            
        cv_splits[f"fold_{fold_num}"] = valid_val_pats
        
    return cv_splits

# ==========================================
#   FUNCIONES DE EVALUACIÓN Y GRÁFICOS
# ==========================================

def plot_confusion_matrix(y_true, y_pred, title, subtitle, filename):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Sano (0)', 'H. pylori (1)'],
                yticklabels=['Sano (0)', 'H. pylori (1)'])
    plt.xlabel('Predicción')
    plt.ylabel('Realidad')
    plt.title(title, fontsize=12, fontweight='bold')
    plt.suptitle(subtitle, fontsize=8, y=0.92)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()
    # print(f"Matriz Confusión guardada: {filename}") # Desactivado para no llenar la consola

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
        print(f"❌ ERROR: Modelo no encontrado: {filename}")
        return None

    model = NeuralNetwork_withAttention(
        input_dim=input_dim, # Usamos la dimensión ajustada (2048 o 512)
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
        print(f"❌ Error cargando pesos de {filename}: {e}")
        return None

def generate_enhanced_roc(roc_logs):
    print("\nGenerando ROC Curve Enhanced...")
    
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    
    plt.figure(figsize=(10, 8))
    
    for i, (fold_name, fold_data) in enumerate(roc_logs.items()):
        labels = fold_data["labels"]
        probs = fold_data["probs"]
        
        if len(np.unique(labels)) < 2: continue # No se puede calcular AUC si solo hay una clase

        fpr, tpr, _ = roc_curve(labels, probs)
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        
        plt.plot(mean_fpr, interp_tpr, lw=1, alpha=0.3, label=f'{fold_name} (AUC = {roc_auc:.3f})')

    if not tprs:
        print("Advertencia: No hay suficientes datos para generar la curva ROC media (menos de 2 clases en todos los folds).")
        plt.close()
        return

    # Calcular Media y Std
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    std_tpr = np.std(tprs, axis=0)

    # Pintar Media (Gruesa)
    plt.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (AUC = %0.3f $\pm$ %0.2f)' % (mean_auc, std_auc), lw=2.5)

    # Pintar Sombra (Std)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.2, label=r'$\pm$ 1 std. dev.')

    # Decoración
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'Cross-Validation ROC Curves - {MODEL_TYPE}', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    plt.savefig(os.path.join(OUTPUT_DIR, "CV_ROC_Enhanced.png"), dpi=300)
    plt.close()
    print(f"Gráfica ROC Enhanced guardada en: CV_ROC_Enhanced.png")


def run_cv_inference_and_evaluate():
    global FIXED_THRESHOLD, INPUT_DIM

    print(f"--- INFERENCIA CV (10 FOLDS) - {MODEL_TYPE} con UMBRAL FIJO {FIXED_THRESHOLD} ---")
    
    df = pd.read_csv(CSV_PATH)
    try:
        cv_splits = load_folds_patients(df, FEATURE_DIR)
    except Exception as e:
        print(f"Error al cargar la distribución de folds: {e}")
        return

    roc_log = {}
    all_results = []
    
    # 2. Iterar sobre los 10 modelos
    for i, model_filename in enumerate(MODELS_FILES):
        fold_num = i + 1
        fold_name = f"fold_{fold_num}"
        
        if fold_name not in cv_splits:
            print(f"\n--- Saltando Fold {fold_num}: No hay pacientes válidos para validación ---")
            continue
            
        print(f"\n--- Evaluando {fold_name}: {model_filename} (T FIJO={FIXED_THRESHOLD:.3f}) ---")
        
        # Cargar modelo
        model = load_model(model_filename, INPUT_DIM)
        if model is None:
            continue

        # Cargar datos de Validación del Fold
        val_pats = cv_splits[fold_name]
        val_ds = PatientFeatureDataset(val_pats, FEATURE_DIR, df)
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

        # 3. Obtener probabilidades
        fold_probs = []
        fold_labels = []
        
        with torch.no_grad():
            for pid, features, label in val_loader:
                try:
                    features = features.to(DEVICE)
                    # La salida es (prob, attention_weights)
                    prob, _ = model(features) 
                    fold_probs.append(prob.item())
                    fold_labels.append(label.item())
                except Exception as e:
                    print(f"Error inferencia en {pid}: {e}")
                    continue
        
        current_labels = np.array(fold_labels)
        fold_probs = np.array(fold_probs)
        
        # 🌟 CLASIFICACIÓN CON UMBRAL FIJO DE 0.5
        preds = (fold_probs >= FIXED_THRESHOLD).astype(int)
        
        # 5. Calcular métricas
        acc = accuracy_score(current_labels, preds)
        try:
            auc_score = roc_auc_score(current_labels, fold_probs)
        except: 
            auc_score = 0.5 
            
        tn, fp, fn, tp = confusion_matrix(current_labels, preds, labels=[0,1]).ravel()
        sens = tp/(tp+fn) if (tp+fn)>0 else 0
        spec = tn/(tn+fp) if (tn+fp)>0 else 0
        
        print(f"  AUC: {auc_score:.4f}, Acc: {acc:.4f}, Sens: {sens:.4f}, Spec: {spec:.4f} ({len(current_labels)} pacientes)")

        all_results.append({
            "Fold": fold_num,
            "Model_File": model_filename,
            "Threshold_Used": FIXED_THRESHOLD,
            "AUC": auc_score,
            "Accuracy": acc,
            "Sensitivity": sens,
            "Specificity": spec,
            "TN": tn, "FP": fp, "FN": fn, "TP": tp
        })
        
        # 6. Guardar ROC logs para la gráfica conjunta
        roc_log[fold_name] = {
            "labels": current_labels,
            "probs": fold_probs,
        }
        
        # 7. Generar Matriz de Confusión Individual (T=0.5)
        plot_confusion_matrix(
            current_labels, preds,
            title=f"Matriz de Confusión - {fold_name} ({MODEL_TYPE})",
            subtitle=f"Model: {model_filename} | Umbral Fijo: {FIXED_THRESHOLD:.1f}\nSet: Validación CV ({len(current_labels)} pacients)",
            filename=f"cm_cv_{MODEL_TYPE}_{fold_name}_T0.5.png"
        )
        
    # 8. Generar Resumen Global
    if all_results:
        df_resumen = pd.DataFrame(all_results)
        output_csv = os.path.join(OUTPUT_DIR, f"cv_predictions_{MODEL_TYPE}_T0.5.csv")
        df_resumen.to_csv(output_csv, index=False)
        print(f"\nResumen de resultados guardado en: {output_csv}")
        print("\n--- Resumen por Fold ---")
        print(df_resumen[['Fold', 'AUC', 'Accuracy', 'Sensitivity', 'Specificity']])

        # 9. Generar ROC conjunta
        generate_enhanced_roc(roc_log)

        # 10. Generar CM total (sumando todos los folds)
        all_labels = np.concatenate([v['labels'] for v in roc_log.values()])
        all_probs = np.concatenate([v['probs'] for v in roc_log.values()])
        all_preds = (all_probs >= FIXED_THRESHOLD).astype(int)
        
        plot_confusion_matrix(
            all_labels, all_preds, 
            title=f"Matriz de Confusión - TOTAL CV ({MODEL_TYPE})", 
            subtitle=f"Suma de 10 Folds | Umbral Fijo: {FIXED_THRESHOLD:.1f}\nTotal Pacientes: {len(all_labels)}",
            filename=f"cm_cv_TOTAL_{MODEL_TYPE}_T0.5.png"
        )
        print(f"Matriz Confusión TOTAL guardada: cm_cv_TOTAL_{MODEL_TYPE}_T0.5.png")


if __name__ == "__main__":
    run_cv_inference_and_evaluate()
    print(f"\nEvaluación finalizada. Resultados en: {OUTPUT_DIR}")