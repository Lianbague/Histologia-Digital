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

# Importar tus modelos y dataset
from S2_MIL.S2_models import NeuralNetwork_withAttention
from S2_HoldOut_test.S2_HoldOut_dataset import HoldOutDataset

# ==========================================
#   CONFIGURACIÓN GLOBAL
# ==========================================

MODEL_TYPE = 'AE' 

# RUTAS BASE
BASE_DIR = '/export/fhome/maed03'
CSV_PATH = '/export/fhome/maed/HelicoDataSet/PatientDiagnosis.csv'
OUTPUT_DIR = os.path.join(BASE_DIR,"S2_HoldOut_test", "S2_HoldOut_final_evaluation")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Determinación de rutas según el modelo
if MODEL_TYPE == 'ResNet':
    FEATURE_DIR = os.path.join(BASE_DIR, "Features_HoldOut_ResNet")
    MODELS_AND_THRESHOLDS = [
        ["S2_best_model_ResNet_fold1.pth",0.606],["S2_best_model_ResNet_fold2.pth",0.428],
        ["S2_best_model_ResNet_fold3.pth",0.759],["S2_best_model_ResNet_fold4.pth",0.625],
        ["S2_best_model_ResNet_fold5.pth",0.295], ["S2_best_model_ResNet_fold6.pth",0.337],
        ["S2_best_model_ResNet_fold7.pth",0.629], ["S2_best_model_ResNet_fold8.pth",0.543],
        ["S2_best_model_ResNet_fold9.pth",0.396], ["S2_best_model_ResNet_fold10.pth",0.517],]

elif MODEL_TYPE == 'Triplet':
    FEATURE_DIR = os.path.join(BASE_DIR, "Features_HoldOut_Triplet128")
    MODELS_AND_THRESHOLDS = [
        ["S2_best_model_Triplet_fold1.pth",0.637],["S2_best_model_Triplet_fold2.pth",0.662],
        ["S2_best_model_Triplet_fold3.pth",0.514],["S2_best_model_Triplet_fold4.pth",0.693],
        ["S2_best_model_Triplet_fold5.pth",0.413], ["S2_best_model_Triplet_fold6.pth",0.819],
        ["S2_best_model_Triplet_fold7.pth",0.724], ["S2_best_model_Triplet_fold8.pth",0.819],
        ["S2_best_model_Triplet_fold9.pth",0.610], ["S2_best_model_Triplet_fold10.pth",0.335],]

elif MODEL_TYPE == 'AE':
    FEATURE_DIR = os.path.join(BASE_DIR, "Features_HoldOut_AE")
    MODELS_AND_THRESHOLDS = [
        ["S2_best_model_AE_fold1.pth",0.753],["S2_best_model_AE_fold2.pth",0.273],
        ["S2_best_model_AE_fold3.pth",0.690],["S2_best_model_AE_fold4.pth",0.740],
        ["S2_best_model_AE_fold5.pth",0.480], ["S2_best_model_AE_fold6.pth",0.826],
        ["S2_best_model_AE_fold7.pth",0.463], ["S2_best_model_AE_fold8.pth",0.508],
        ["S2_best_model_AE_fold9.pth",0.217], ["S2_best_model_AE_fold10.pth",0.564],]
else:
    raise ValueError(f"MODEL_TYPE '{MODEL_TYPE}' no reconocido.")

OUTPUT_PRED_CSV = os.path.join(OUTPUT_DIR, f"holdout_predictions_{MODEL_TYPE}.csv")

print(f"--- CONFIGURACION ---")
print(f"Modelo: {MODEL_TYPE}")
print(f"Features: {FEATURE_DIR}")
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
        print(f"❌ ERROR: Modelo no encontrado: {filename}")
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
        print(f"❌ Error cargando pesos de {filename}: {e}")
        return None

# =========================================================================
#  FUNCIÓN MODIFICADA PARA ROC PROMEDIO + TABLA
# =========================================================================
def generate_individual_roc_with_thresholds(true_labels, fold_probs_list, thresholds_data, model_type, results_summary):
    print("\nGenerando ROC Curves Individuales con Promedio y Tabla...")
    roc_dir = os.path.join(OUTPUT_DIR, "Individual_ROC_Plots")
    os.makedirs(roc_dir, exist_ok=True)
    
    # Crear figura con 2 espacios: Gráfico a la izquierda, Tabla a la derecha
    fig = plt.figure(figsize=(16, 8))
    # width_ratios controla el ancho relativo: 3 partes para el gráfico, 1 parte para la tabla
    gs = fig.add_gridspec(1, 2, width_ratios=[3, 1.2]) 
    
    ax_roc = fig.add_subplot(gs[0, 0])
    ax_table = fig.add_subplot(gs[0, 1])
    
    # Variables para calcular el promedio (Mean ROC)
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    aucs = []
    
    colors = plt.cm.get_cmap('tab10', len(fold_probs_list)) 
    
    # --- PLOTEADO DE MODELOS INDIVIDUALES ---
    for i, (fold_probs, model_data) in enumerate(zip(fold_probs_list, thresholds_data)):
        fold_num = i + 1
        model_filename, threshold = model_data
        
        # Calcular ROC
        fpr, tpr, thresholds_roc = roc_curve(true_labels, fold_probs)
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        
        # Interpolar TPR para el cálculo del promedio
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        
        # Encontrar punto del umbral
        threshold_diffs = np.abs(thresholds_roc - threshold)
        optimal_idx = np.argmin(threshold_diffs)

        # Plotear curva del modelo (MÁS FINA y TRANSPARENTE)
        ax_roc.plot(fpr, tpr, color=colors(i), lw=1.2, alpha=0.4, 
                  label=f'Fold {fold_num} (AUC = {roc_auc:.3f})')
        
        # Marcar umbral
        ax_roc.plot(fpr[optimal_idx], tpr[optimal_idx], 'o', 
                  color=colors(i), markersize=5, alpha=0.6)

    # --- CÁLCULO Y PLOT DEL PROMEDIO (NEGRO) ---
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)

    ax_roc.plot(mean_fpr, mean_tpr, color='black', label=f'Mean ROC (AUC = {mean_auc:.3f} $\pm$ {std_auc:.2f})',
              lw=3, alpha=1) # Línea negra gruesa

    # Decoración del gráfico ROC
    ax_roc.plot([0, 1], [0, 1], linestyle='--', lw=2, color='gray', label='Chance', alpha=.8)
    ax_roc.set_xlim([-0.01, 1.01])
    ax_roc.set_ylim([-0.01, 1.01])
    ax_roc.set_xlabel('False Positive Rate', fontsize=12)
    ax_roc.set_ylabel('True Positive Rate', fontsize=12)
    ax_roc.set_title(f'ROC Curves & Performance Table - {model_type}', fontsize=14, fontweight='bold')
    ax_roc.legend(loc="lower right", fontsize=9)
    ax_roc.grid(True, alpha=0.3)
    
    # --- CONSTRUCCIÓN DE LA TABLA A LA DERECHA ---
    ax_table.axis('off') # Quitar ejes del subplot de la tabla
    
    # Preparar datos: Acc, Sens, Spec, AUC
    # Asumimos que results_summary tiene el mismo orden que thresholds_data
    table_data = []
    cols = ["Fold", "Acc", "Sens", "Spec", "AUC"]
    
    # Valores individuales
    for res in results_summary:
        row = [
            f"F{res['Fold']}",
            f"{res['Accuracy']:.3f}",
            f"{res['Sensitivity']:.3f}",
            f"{res['Specificity']:.3f}",
            f"{res['AUC']:.3f}"
        ]
        table_data.append(row)
        
    # Añadir fila de promedio al final
    df_temp = pd.DataFrame(results_summary)
    mean_row = [
        "MEAN",
        f"{df_temp['Accuracy'].mean():.3f}",
        f"{df_temp['Sensitivity'].mean():.3f}",
        f"{df_temp['Specificity'].mean():.3f}",
        f"{mean_auc:.3f}"
    ]
    table_data.append(mean_row)

    # Crear la tabla
    the_table = ax_table.table(
        cellText=table_data,
        colLabels=cols,
        loc='center',
        cellLoc='center'
    )
    
    # Ajustar estilo de tabla
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(10)
    the_table.scale(1, 1.8) # Hacer filas más altas
    
    # Destacar la fila del promedio (negrita o color de fondo)
    last_row_index = len(table_data) - 1
    for k, cell in the_table.get_celld().items():
        row_idx, col_idx = k
        if row_idx == 0: # Headers
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#e6e6e6')
        elif row_idx == last_row_index + 1: # Fila de media (row index es +1 por el header)
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#f2f2f2')

    plt.tight_layout()
    filename = f'holdout_roc_average_table_{model_type}.png'
    plt.savefig(os.path.join(roc_dir, filename), dpi=300)
    plt.close()
    print(f"Gráfica ROC con Promedio y Tabla guardada: {filename}")


def run_individual_evaluation():
    print(f"--- INICIANDO EVALUACIÓN INDIVIDUAL (10 MODELS) - {MODEL_TYPE} ---")
    
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
    
    print("Leyendo etiquetas reales...")
    for pid, features, label in loader:
        try:
            true_labels.append(label.item())
        except: continue
            
    true_labels = np.array(true_labels)

    all_results = []
    all_fold_probs = [] 

    # 2. Iterar sobre los 10 modelos
    for i, (model_filename, threshold) in enumerate(MODELS_AND_THRESHOLDS):
        fold_num = i + 1
        print(f"\n--- Evaluando Modelo {fold_num}: {model_filename} (T={threshold:.3f}) ---")
        
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
                    continue
        
        fold_probs = np.array(fold_probs)
        
        if len(fold_probs) != len(true_labels):
            print(f"⚠️ Warning: Longitud de probs ({len(fold_probs)}) != labels ({len(true_labels)})")
            min_len = min(len(fold_probs), len(true_labels))
            fold_probs = fold_probs[:min_len]
            current_labels = true_labels[:min_len]
        else:
            current_labels = true_labels

        all_fold_probs.append(fold_probs)
        
        # 4. Clasificar usando el umbral específico
        preds = (fold_probs >= threshold).astype(int)
        
        # 5. Calcular métricas para el resumen
        acc = accuracy_score(current_labels, preds)
        try:
            auc_score = roc_auc_score(current_labels, fold_probs)
        except: auc_score = 0.5 
            
        tn, fp, fn, tp = confusion_matrix(current_labels, preds, labels=[0,1]).ravel()
        sens = tp/(tp+fn) if (tp+fn)>0 else 0
        spec = tn/(tn+fp) if (tn+fp)>0 else 0
        
        print(f"  AUC: {auc_score:.4f}, Acc: {acc:.4f}, Sens: {sens:.4f}, Spec: {spec:.4f}")

        all_results.append({
            "Fold": fold_num,
            "Model_File": model_filename,
            "Threshold": threshold,
            "AUC": auc_score,
            "Accuracy": acc,
            "Sensitivity": sens,
            "Specificity": spec,
        })
        
        # 6. Generar Matriz de Confusión
        plot_confusion_matrix(
            current_labels, preds,
            title=f"Matriu de Confusió - Fold {fold_num} ({MODEL_TYPE})",
            subtitle=f"Model: {model_filename} | Opt. Threshold: {threshold:.3f}\nSet: HoldOut ({len(current_labels)} pacients)",
            filename=f"cm_holdout_{MODEL_TYPE}_fold{fold_num}_T{threshold:.3f}.png"
        )
        
    # 7. Guardar Resumen de Resultados
    if all_results:
        df_resumen = pd.DataFrame(all_results)
        df_resumen.to_csv(OUTPUT_PRED_CSV, index=False)
        print(f"\nResumen de resultados guardado en: {OUTPUT_PRED_CSV}")
        print("\nResumen por Fold:")
        print(df_resumen)

        # Generar ROC conjunta CON PROMEDIO Y TABLA
        generate_individual_roc_with_thresholds(
            true_labels, 
            all_fold_probs, 
            MODELS_AND_THRESHOLDS, 
            MODEL_TYPE,
            all_results # <-- Pasamos los resultados aquí
        )

if __name__ == "__main__":
    run_individual_evaluation()