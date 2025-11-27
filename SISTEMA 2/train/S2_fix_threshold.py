from S2_MIL.S2_system_evaluation import generate_roc_graphics

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import torchvision.models as models
import pandas as pd
import os
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, f1_score, roc_curve
import matplotlib.pyplot as plt
import pickle
import glob
from PIL import Image


from S2_MIL.S2_train_with_logs import train_system2
from S2_MIL.S2_train_10_folds_with_logs import train_system2_10folds
from S2_MIL.S2_models import NeuralNetwork_withAttention
from S2_MIL.S2_train_with_logs import PatientFeatureDataset

# !!!! ELEGIR EL MEJOR FOLD O HACER EL PROMEDIO DE LOS 5 !!!!!!!!!!!
CSV_PATH = '/export/fhome/maed/HelicoDataSet/PatientDiagnosis.csv'  # Path al CSV amb les etiquetes dels pacients HoldOut
FEATURES_DIR = "/export/fhome/maed03/Features_ResNet"  # Path a les característiques extretes dels pacients HoldOut
MODEL_PREFIX = 'S2_best_model_ResNet_fold'
NUM_FOLDS = 5
BATCH_SIZE = 1
OUTPUT_ROC_PLO_WITH_THRESHOLDS = "roc_curves_all_fold_with_thresholds.png"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(input_dim, MODEL_PATH):
    model = NeuralNetwork_withAttention(
        input_dim=input_dim,
        project_dim=512,
        decom_space=128,
        attention_branches=1,
        attention_type="GatedAttention"
    ).to(DEVICE)

    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model


def evaluate_folds():
    df = pd.read_csv(CSV_PATH)
    available_pats = [f.replace('.pt', '') for f in os.listdir(FEATURES_DIR) if f.endswith('.pt')]
    valid_patients = [p for p in available_pats if p in df['CODI'].values]
    
    # Determine INPUT_DIM once
    temp_feat, _ = PatientFeatureDataset([valid_patients[0]], FEATURES_DIR, df)[0]
    INPUT_DIM = temp_feat.shape[1]

    results = dict()

    # Bucle K-Fold
    kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
    for fold, (train_idx, val_idx) in enumerate(kf.split(valid_patients)):
        fold_num = fold + 1
        print(f"Processing Fold {fold_num}/{NUM_FOLDS}")
        
        val_pats = [valid_patients[i] for i in val_idx]
        val_ds = PatientFeatureDataset(val_pats, FEATURES_DIR, df)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
        
        # Construct the correct model path for the current fold
        model_path = f"{MODEL_PREFIX}{fold_num}.pth"
        if not os.path.exists(model_path):
            print(f"Error: Model not found at {model_path}. Skipping fold.")
            continue
            
        # Load the model for this fold
        model = load_model(INPUT_DIM, model_path)

        model.eval()
        val_probs = []
        val_labels = []
            
        with torch.no_grad():
            for features, label in val_loader:
                features = features.to(DEVICE)
                probs, _ = model(features)
                val_probs.append(probs.item())
                val_labels.append(label.item())
            
        try:
            auc = roc_auc_score(val_labels, val_probs)
        except ValueError: # Catch error for constant labels
            auc = 0.5 
            
        results[f"fold_{fold_num}"] = {
            "labels": val_labels,
            "probs": val_probs,
            "auc": auc
        } 
    return results

def plot_rocs_with_optimal_thresholds(labels_and_probs_per_fold):
    
    plt.figure(figsize=(9, 8))
    
    for fold_name, data in labels_and_probs_per_fold.items():
        labels = data['labels']
        probs = data['probs']
        
        # 1. Calculate ROC curve metrics
        fpr, tpr, thresholds = roc_curve(labels, probs)
        auc = data['auc'] # Use the AUC calculated in evaluate_folds
        
        # 2. Find the Optimal Threshold (Point closest to (0, 1) or Youden's J)
        # Using Youden's J: maximize Sensitivity - (1 - Specificity)
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        optimal_threshold = thresholds[optimal_idx]
        optimal_fpr = fpr[optimal_idx]
        optimal_tpr = tpr[optimal_idx]

        # 3. Plot the ROC curve for the current fold
        plt.plot(fpr, tpr, 
                 label=f"{fold_name} (AUC = {auc:.4f})",
                 alpha=0.8)
        
        # 4. Plot the optimal threshold point
        plt.plot(optimal_fpr, optimal_tpr, 
                 marker='o', markersize=5, 
                 color=plt.gca().lines[-1].get_color(), # Use same color as the line
                 linestyle='None',
                 label=f"Opt. Thresh. {optimal_threshold:.2f} ({fold_name})")

    # 5. Add the diagonal chance line
    plt.plot([0, 1], [0, 1], 'k--', label="Chance (AUC=0.5)")
    
    # 6. Finalize plot
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
    plt.ylabel("True Positive Rate (Sensitivity)", fontsize=12)
    plt.title("ROC Curves and Optimal Thresholds per K-Fold Split", fontsize=14)
    plt.legend(loc='lower right', fontsize=9)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_ROC_PLO_WITH_THRESHOLDS, dpi=150)
    plt.close()

    print(f"\nROC curves with optimal thresholds saved to {OUTPUT_ROC_PLO_WITH_THRESHOLDS}")


if __name__ == "__main__":

    labels_and_probs_per_fold = evaluate_folds()
    plot_rocs_with_optimal_thresholds(labels_and_probs_per_fold)