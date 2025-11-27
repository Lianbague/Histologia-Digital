import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, auc
from scipy.interpolate import PchipInterpolator

# Importamos la función de entrenamiento que devuelve los logs
from S2_MIL.S2_train_10_folds_with_logs import train_system2_10folds

# --- CONFIGURACIÓN ---
MODEL_TYPE = 'AE' # Elige: 'ResNet', 'AE', 'Triplet'
OUTPUT_DIR = f"/export/fhome/maed03/S2_MIL/Evaluation_{MODEL_TYPE}/"

def plot_confusion_matrix(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Sano (0)', 'H. pylori (1)'],
                yticklabels=['Sano (0)', 'H. pylori (1)'])
    plt.xlabel('Predicción')
    plt.ylabel('Realidad')
    plt.title(title, fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()

def generate_loss_graphics(loss_logs):
    print("Generando gráficas de Loss...")
    loss_dir = os.path.join(OUTPUT_DIR, "Loss_Plots")
    os.makedirs(loss_dir, exist_ok=True)
    
    for fold_name, fold_data in loss_logs.items():
        losses = fold_data["train"]
        plt.figure(figsize=(8, 5))
        plt.plot(losses, label=f'{fold_name}', marker='o', markersize=3)
        plt.title(f'Training Loss - {fold_name}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        filename = f'loss_{fold_name}.png'
        plt.savefig(os.path.join(loss_dir, filename), dpi=150)
        plt.close()

def generate_enhanced_roc(roc_logs):
    print("Generando ROC Curve Enhanced...")
    
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    
    plt.figure(figsize=(10, 8))
    
    # Iterar por cada fold
    for i, (fold_name, fold_data) in enumerate(roc_logs.items()):
        labels = fold_data["labels"]
        probs = fold_data["probs"]
        
        # Calcular ROC del fold
        fpr, tpr, _ = roc_curve(labels, probs)
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        
        # Interpolar para la media
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        
        # Pintar línea fina del fold
        plt.plot(mean_fpr, interp_tpr, lw=1, alpha=0.3, label=f'{fold_name} (AUC = {roc_auc:.3f})')

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


def generate_roc_with_threshold(roc_logs):
    print("Generando ROC Curves Individuales con Umbral Óptimo...")
    roc_dir = os.path.join(OUTPUT_DIR, "ROC_Threshold_Plots")
    os.makedirs(roc_dir, exist_ok=True)
    
    optimal_thresholds = {}
    
    plt.figure(figsize=(10, 8))
    
    # Colores para distinguir los folds
    colors = plt.cm.get_cmap('hsv', len(roc_logs)) 
    
    # Iterar por cada fold
    for i, (fold_name, fold_data) in enumerate(roc_logs.items()):
        labels = fold_data["labels"]
        probs = fold_data["probs"]
        
        # 1. Calcular ROC
        fpr, tpr, thresholds = roc_curve(labels, probs)
        roc_auc = auc(fpr, tpr)
        
        # 2. Encontrar el Umbral Óptimo (Distancia euclidiana mínima a (0, 1))
        # Distancia = sqrt((1 - TPR)^2 + (FPR - 0)^2)
        distances = np.sqrt((1 - tpr)**2 + fpr**2)
        optimal_idx = np.argmin(distances)
        optimal_threshold = thresholds[optimal_idx]
        
        optimal_thresholds[fold_name] = optimal_threshold
        
        # 3. Plotear la curva ROC del fold
        plt.plot(fpr, tpr, color=colors(i), lw=1.5, alpha=0.8, 
                 label=f'{fold_name} (AUC = {roc_auc:.3f})')
        
        # 4. Marcar el Umbral Óptimo en el gráfico
        plt.plot(fpr[optimal_idx], tpr[optimal_idx], 'X', color=colors(i), markersize=8, 
                 label=f'Opt. Threshold ({optimal_threshold:.3f})')
        
    # 5. Decoración (Se aplica una sola vez para todas las curvas)
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'Individual ROC Curves and Optimal Thresholds - {MODEL_TYPE}', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=8)
    plt.grid(True, alpha=0.3)
    
    filename = 'CV_ROC_Individual_Thresholds.png'
    plt.savefig(os.path.join(roc_dir, filename), dpi=300)
    plt.close()
    
    print(f"  Umbrales óptimos calculados: {optimal_thresholds}")
    return optimal_thresholds

"""
def generate_confusion_matrices(roc_logs):
    print("Generando Matrices de Confusión...")
    cm_dir = os.path.join(OUTPUT_DIR, "Confusion_Matrices")
    os.makedirs(cm_dir, exist_ok=True)
    
    all_labels = []
    all_preds = []
    
    for fold_name, fold_data in roc_logs.items():
        labels = np.array(fold_data["labels"])
        probs = np.array(fold_data["probs"])
        preds = (probs >= 0.5).astype(int) # Threshold 0.5 por defecto
        
        # Guardar para Global
        all_labels.extend(labels)
        all_preds.extend(preds)
        
        # Plot Individual
        plot_confusion_matrix(labels, preds, f"Confusion Matrix - {fold_name}", f"cm_{fold_name}.png")
        
    # Plot Global (Suma de todos los folds)
    plot_confusion_matrix(all_labels, all_preds, f"Confusion Matrix - TOTAL ({len(all_labels)} pacientes)", "cm_TOTAL.png")
"""

def generate_confusion_matrices(roc_logs, optimal_thresholds): # <--- Acepta umbrales
    print("Generando Matrices de Confusión (con umbrales óptimos)...")
    cm_dir = os.path.join(OUTPUT_DIR, "Confusion_Matrices")
    os.makedirs(cm_dir, exist_ok=True)
    
    all_labels = []
    all_preds = []
    
    for fold_name, fold_data in roc_logs.items():
        labels = np.array(fold_data["labels"])
        probs = np.array(fold_data["probs"])
        
        # 🌟 Usar el umbral óptimo específico del fold
        if fold_name in optimal_thresholds:
            threshold = optimal_thresholds[fold_name]
        else:
            # Usar 0.5 como fallback si no se encuentra
            threshold = 0.5 
            print(f"  Advertencia: Umbral no encontrado para {fold_name}. Usando 0.5.")
            
        preds = (probs >= threshold).astype(int) 
        
        # Guardar para Global
        all_labels.extend(labels)
        all_preds.extend(preds)
        
        # Plot Individual
        title = f"CM - {fold_name} (T={threshold:.3f})"
        filename = f"cm_{fold_name}_T{threshold:.3f}.png"
        plot_confusion_matrix(labels, preds, title, filename)
        
    # Plot Global (Suma de todos los folds)
    plot_confusion_matrix(all_labels, all_preds, 
                          f"CM - TOTAL ({len(all_labels)} muestras, T_i óptimos)", 
                          "cm_TOTAL_OptimalT.png")

if __name__ == "__main__":
    # 1. Ejecutar Entrenamiento y Obtener Logs
    print(f"--- EVALUANDO SISTEMA 2: {MODEL_TYPE} ---")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    loss_log, roc_log = train_system2_10folds(MODEL_TYPE)

    # 2. Generar Gráficas
    generate_loss_graphics(loss_log)
    generate_enhanced_roc(roc_log)
    optimal_thresholds = generate_roc_with_threshold(roc_log)
    # TODO: Fins optimal thresholds
    generate_confusion_matrices(roc_log, optimal_thresholds)
    
    print(f"Evaluación finalizada. Resultados en: {OUTPUT_DIR}")