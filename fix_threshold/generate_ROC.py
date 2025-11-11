# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import sys
import pandas as pd
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
import time
import re

# Assuming ae_models.py (with AutoEncoderCNN and AEConfigs) is accessible
from ae_models import AutoEncoderCNN, AEConfigs 

# --- Model and Transform Functions (from your provided code) ---

def get_eval_transforms():
    """ Uses the same transformations and normalization as training. """
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def calculate_reconstruction_error(image_path, model, device):
    """ Loads a patch and calculates its reconstruction error. """
    if not os.path.exists(image_path):
        # Using print(f"ERROR: ...") as in original code, but we'll return None 
        # and handle the skip in the main loop to keep the script clean.
        print(f"DEBUG ERROR: Image not found at the generated path: {image_path}")
        return None
        
    transforms_eval = get_eval_transforms()
    
    # Load image
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"ERROR: Could not load image at {image_path}. Skipping. Error: {e}")
        return None

    input_tensor = transforms_eval(image).unsqueeze(0).to(device) 

    # Reconstruction
    model.eval()
    with torch.no_grad():
        reconstruction = model(input_tensor)
        
    # Calculate Reconstruction Error (MSE / L2 Loss) - Mean over all dimensions
    # We use reduction='none' and then mean() to ensure we get a single scalar error per image.
    l_red = nn.MSELoss(reduction='none')(reconstruction, input_tensor).mean(dim=[1, 2, 3])
    
    return l_red.item()

# --------------------------------------------------------------------------------

def main():
    """ Main function to load data, process images, and generate the ROC curve. """
    
    # --- Configuration ---
    MODEL_SAVE_PATH = 'autoencoder_negativa_best.pth'
    CONFIG = '1'
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # --- File Paths ---
    # CSV file containing the list of images and their ground-truth labels
    CSV_PATH = 'threshold_set_balanced.csv' 
    # Base directory where the annotated images are located (adjust if needed)
    BASE_IMAGE_DIR = '/export/fhome/maed/HelicoDataSet/CrossValidation/Annotated' 
    ROC_CURVE_SAVE_PATH = 'roc_curve.png' # Path to save the ROC curve plot
    
    if not os.path.exists(MODEL_SAVE_PATH):
        print(f"ERROR: Model file not found at {MODEL_SAVE_PATH}. Exiting.")
        sys.exit(1)
    
    if not os.path.exists(CSV_PATH):
        print(f"ERROR: CSV file not found at {CSV_PATH}. Exiting.")
        sys.exit(1)
        
    # --- 1. Load Model ---
    print(f"✨ Loading trained model weights from {MODEL_SAVE_PATH} to {DEVICE}...")
    config = AEConfigs(config_id=CONFIG, input_channels=3)
    model = AutoEncoderCNN(
        net_paramsEnc=config.net_paramsEnc, 
        inputmodule_paramsDec=config.inputmodule_paramsDec, 
        net_paramsDec=config.net_paramsDec
    )
    
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    print("✅ Model loaded successfully.")
    
    # --- 2. Load Data and Prepare Lists ---
    print(f"\n📚 Loading data from {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)

    # Initialize lists to store results
    reconstruction_errors = []
    true_labels = []

    # Map Presence: -1 (Negative/Healthy) -> 0 (Negative Class)
    #              1 (Positive/Anomaly) -> 1 (Positive Class)
    # ROC curve expects 0 for negative and 1 for positive.
    df['GroundTruth'] = df['Presence'].apply(lambda x: 1 if x == 1 else 0)
    
    total_images = len(df)
    print(f"Total images to process: {total_images}")
    start_time = time.time()
    
    # --- 3. Iterate, Process, and Calculate Errors ---
    print("\n🔍 Starting image processing and error calculation...")
    
    for index, row in df.iterrows():
        pat_section_folder = row['Pat_Section'] # e.g., 'B22-129_0'
        window_id = row['Window_ID']           # e.g., '10'


        # Extract numeric part and augmentation suffix if present
        match = re.match(r"(\d+)(_Aug\d*)?$", str(window_id))
        if not match:
            print(f"Skipping row {index}: Window_ID '{window_id}' does not match expected pattern.")
            continue

        num_part = int(match.group(1))
        aug_part = match.group(2) if match.group(2) else ""

        # Pad number to 4 digits and reconstruct filename
        image_filename = f"{num_part:05d}{aug_part}.png"

        image_path = os.path.join(BASE_IMAGE_DIR, pat_section_folder, image_filename)
        
        # Calculate the reconstruction error
        error = calculate_reconstruction_error(image_path, model, DEVICE)
        
        if error is not None:
            reconstruction_errors.append(error)
            true_labels.append(row['GroundTruth'])
        
        # Simple progress update
        if (index + 1) % 100 == 0 or (index + 1) == total_images:
            elapsed = time.time() - start_time
            rate = (index + 1) / elapsed
            sys.stdout.write(f"\rProcessed {index + 1}/{total_images} images | Rate: {rate:.2f} img/sec")
            sys.stdout.flush()

    end_time = time.time()
    print(f"\n\nProcessing complete! Time taken: {end_time - start_time:.2f} seconds.")
    
    if not reconstruction_errors:
        print("ERROR: No images were successfully processed. Cannot generate ROC curve.")
        sys.exit(1)

    # --- Code to be inserted after the image processing loop (Step 3) ---

    # Create a DataFrame from the collected results
    results_df = pd.DataFrame({
        'Error': reconstruction_errors,
        # Convert GroundTruth back to original Presence values for clarity
        # 0 -> -1 (Healthy), 1 -> 1 (Sick)
        'Presence': [1 if label == 1 else -1 for label in true_labels] 
    })
    
    # Calculate the mean error for each group
    mean_error_sick = results_df[results_df['Presence'] == 1]['Error'].mean()
    mean_error_healthy = results_df[results_df['Presence'] == -1]['Error'].mean()
    
    print("\n--- Mean Reconstruction Error Analysis ---")
    print(f"Mean Error (Sick Images, Presence = 1): **{mean_error_sick:.6f}**")
    print(f"Mean Error (Healthy Images, Presence = -1): **{mean_error_healthy:.6f}**")
    
    if mean_error_sick > mean_error_healthy:
        ratio = mean_error_sick / mean_error_healthy
        print(f"\n✅ Anomaly Detection Success: The mean error for sick patches is {ratio:.2f}x greater than for healthy patches.")
    else:
        print("\n⚠️ Warning: Mean Sick Error is not greater than Healthy Error. The model may not be effective.")
    
        
    # --- 4. Generate ROC Curve ---
    print("\n📈 Generating ROC Curve...")
    
    # The reconstruction error is the 'score' or 'probability' for the positive class (anomaly). 
    # Higher error -> Higher likelihood of anomaly (Presence=1).
    true_labels_array = np.array(true_labels)
    scores_array = np.array(reconstruction_errors)

    # Calculate the ROC curve points
    fpr, tpr, thresholds = roc_curve(true_labels_array, scores_array)
    # Calculate the Area Under the Curve (AUC)
    roc_auc = auc(fpr, tpr)
    
    print(f"Area Under the Curve (AUC): {roc_auc:.4f}")

    # --- 5. Plot and Save ROC Curve ---
    plt.figure(figsize=(8, 8))
    plt.plot(
        fpr, tpr, color='darkorange', lw=2, 
        label=f'ROC curve (AUC = {roc_auc:.4f})'
    )
    # Plot the diagonal line (random classifier)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    
    plt.savefig(ROC_CURVE_SAVE_PATH)
    print(f"🖼️ ROC Curve saved to {ROC_CURVE_SAVE_PATH}")
    print("\n--- Next Steps ---")
    print("The 'thresholds' array contains the error values that generated each point on the curve.")
    print("You can now analyze the trade-off between TPR and FPR to select the optimal threshold.")


if __name__ == '__main__':
    main()