import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob

from S2_MIL.S2_models import NeuralNetwork_withAttention

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Ajusta estas rutas si es necesario (ej. si usas Triplet o ResNet)
FEATURES_DIR = '/export/fhome/maed03/Features_ResNet'  
IMAGES_DIR = '/export/fhome/maed/HelicoDataSet/CrossValidation/Cropped'    
MODEL_DIR = '/export/fhome/maed03' # Directorio donde están los .pth

# Configuración del Modelo 
INPUT_DIM = 2048 # IMPORTANTE: 128 si usas Triplet, 2048 si es ResNet 
PROJECT_DIM = 512
DECOM_SPACE = 128

def load_model(fold_num):
    """Carga el modelo de un fold específico"""
    model_path = os.path.join(MODEL_DIR, f'S2_best_model_ResNet_fold{fold_num}.pth')
    
    if not os.path.exists(model_path):
        print(f"Aviso: No se encontró {model_path}. Saltando...")
        return None

    model = NeuralNetwork_withAttention(
        input_dim=INPUT_DIM,
        project_dim=PROJECT_DIM,
        decom_space=DECOM_SPACE,
        attention_branches=1,
        attention_type='GatedAttention'
    ).to(DEVICE)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()
        return model
    except Exception as e:
        print(f"Error cargando modelo {fold_num}: {e}")
        return None

def get_patient_data(patient_id):
    """Carga features e imágenes una sola vez"""
    # 1. Features
    feat_path = os.path.join(FEATURES_DIR, f"{patient_id}.pt")
    if not os.path.exists(feat_path):
        print(f"Error: No hay features para {patient_id}")
        return None, None
    
    features = torch.load(feat_path, map_location=DEVICE, weights_only=True)

    # 2. Imágenes (Paths ordenados)
    pat_sections = glob.glob(os.path.join(IMAGES_DIR, f"{patient_id}_*"))
    all_patches_paths = []
    for sec in pat_sections:
        patches = glob.glob(os.path.join(sec, '*.png'))
        all_patches_paths.extend(patches)
        
    if len(all_patches_paths) != features.shape[0]:
        print(f"ALERTA: Dimensiones no coinciden ({len(all_patches_paths)} imgs vs {features.shape[0]} vecs)")
        
    return features, all_patches_paths

def plot_attention(patient_id, image_paths, weights, title, output_filename, top_k=5):
    """
    Pinta 2 filas:
    - Arriba: Los parches con MAYOR atención (Sospechosos de H. pylori)
    - Abajo: Los parches con MENOR atención (Sano o Fondo)
    """
    
    # 1. Identificar índices
    # Top (Mayor a menor)
    top_indices = weights.argsort()[-top_k:][::-1]
    top_values = weights[top_indices]
    
    # Bottom (Menor a mayor - lo que el modelo ignora)
    bot_indices = weights.argsort()[:top_k]
    bot_values = weights[bot_indices]
    
    print(f"  > Generando gráfica High/Low para: {title}")

    # Crear figura con 2 filas
    fig, axes = plt.subplots(2, top_k, figsize=(15, 7))
    fig.suptitle(f"Paciente {patient_id} - {title}\nArriba: Alta Atención (Bacteria) | Abajo: Baja Atención (Sano/Fondo)", fontsize=14)
    
    # --- FILA 1: HIGH ATTENTION ---
    for i, idx in enumerate(top_indices):
        ax = axes[0, i] # Fila 0
        if idx < len(image_paths):
            try:
                img = Image.open(image_paths[idx])
                ax.imshow(img)
                ax.set_title(f"HIGH: {top_values[i]:.4f}\nIdx: {idx}", color='darkred', fontsize=10, fontweight='bold')
            except: ax.text(0.5, 0.5, "Err", ha='center')
        else: ax.text(0.5, 0.5, "N/A", ha='center')
        ax.axis('off')

    # --- FILA 2: LOW ATTENTION ---
    for i, idx in enumerate(bot_indices):
        ax = axes[1, i] # Fila 1
        if idx < len(image_paths):
            try:
                img = Image.open(image_paths[idx])
                ax.imshow(img)
                ax.set_title(f"LOW: {bot_values[i]:.4f}\nIdx: {idx}", color='darkgreen', fontsize=10)
            except: ax.text(0.5, 0.5, "Err", ha='center')
        else: ax.text(0.5, 0.5, "N/A", ha='center')
        ax.axis('off')
        
    plt.tight_layout()
    plt.savefig(output_filename)
    plt.close()
    print(f"    -> Guardado: {output_filename}")

def main(patient_id):
    print(f"\n=== Procesando Paciente: {patient_id} ===")
    
    # 1. Cargar datos (Optimización: Solo se hace una vez)
    features, img_paths = get_patient_data(patient_id)
    if features is None: return

    # Acumulador para el promedio (Ensemble)
    accumulated_weights = np.zeros(features.shape[0])
    models_count = 0

    # 2. Bucle por los 5 Folds (Genera las gráficas individuales)
    for fold in range(1, 6):
        model = load_model(fold)
        if model:
            with torch.no_grad():
                # Inferencia
                _, A = model(features.unsqueeze(0))
                weights = A.squeeze().cpu().numpy()
            
            # Acumular para promedio
            accumulated_weights += weights
            models_count += 1
            
            # Generar gráfica individual
            plot_attention(patient_id, img_paths, weights, 
                           title=f"Modelo Fold {fold}", 
                           output_filename=f"Att_{patient_id}_Fold{fold}.png")
    
    # 3. Generar Gráfica Ensemble (Promedio)
    if models_count > 0:
        avg_weights = accumulated_weights / models_count
        plot_attention(patient_id, img_paths, avg_weights, 
                       title="ENSEMBLE (Promedio 5 Folds)", 
                       output_filename=f"Att_{patient_id}_ENSEMBLE.png")
    else:
        print("No se pudieron cargar modelos para hacer el ensemble.")

if __name__ == '__main__':
    # PACIENTE POSITIVO DE TU DATASET HOLDOUT
    TARGET_PATIENT = 'B22-86' 
    #pacientes holdout'B22-222', B22-85
    # 'B22-86', B22-101 ALTA POSITIU DE ANNOTATED
    main(TARGET_PATIENT)