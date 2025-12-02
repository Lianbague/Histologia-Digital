# Digital Histology: Anomaly Detection and Sample Classification for H. pylori Diagnosis

**Authors:** Adriana, Lian, Martina, Paula
**Institution:** UAB (Universitat Autònoma de Barcelona)
**Project:** Artificial Intelligence in Health
**Goal:** Automate the diagnosis of Helicobacter pylori in gastric biopsies using deep learning techniques to overcome the limitations of manual visual inspection (large images, low bacteria density, subjectivity).

## 1. Project Overview
Helicobacter pylori is a class 1 carcinogen linked to chronic gastritis and stomach cancer. Diagnosis via Whole Slide Images (WSI) is challenging due to the bacteria's tiny size and sparse distribution.
This project implements and compares two distinct Deep Learning systems:
1. **System 1 (Unsupervised):** An Anomaly Detector based on Autoencoders (AE/VAE) trained only on healthy tissue.
2. **System 2 (Supervised):** A Multiple Instance Learning (MIL) classifier with Gated Attention, utilizing feature extraction (ResNet50, Triplet Loss) to diagnose patients from bags of image patches.

## 2. Dataset
The dataset (provided by **QUIRON Salut**) consists of high-resolution histological images of gastric biopsies divided into patches.

| Set | Subset / Folder | Purpose | Description |
| :--- | :--- | :--- | :--- |
| **Cross-Validation** | `Annotated` | **Phase 1 (Patch Level)** | Expert-annotated patches (bacteria/healthy) used to calibrate optimal error thresholds (System 1) or train contrastive learning (System 2). |
| | `Cropped` | **Phase 2 (Patient Level)** | Thousands of patches per patient. Used to calculate Patient Percentage Positive (PPP) or train MIL classifiers. |
| **HoldOut** | - | **Final Testing** | 116 unseen patients reserved strictly for final evaluation of generalization capability. |

> **Note:** For System 1 training, the `Cropped` dataset was filtered to include **only healthy patients (Negative)**.

---

## 3. Methodology & Architecture

The workflow addresses the high resolution of WSIs by processing them as collections of small patches.

### **System 1: Anomaly Detection (Unsupervised)**
*Hypothesis:* A generative model trained on healthy tissue will fail to reconstruct the "anomaly" (bacteria), resulting in a high reconstruction error.

* **Models:**
    * **AutoEncoder (AE):** Trained with L1 Loss. Tested on HSV color space (Hue channel) to isolate bacteria color.
    * **Variational AutoEncoder (VAE):** Probabilistic latent space. Trained with MSE + KLD Loss on RGB images.
* **Metric:** We compared Mean Squared Error (MSE) vs. **99th Percentile (P99)** error. P99 proved superior for detecting small bacteria without dilution.
* **Workflow:**
    1.  Train AE/VAE on healthy patches.
    2.  Define optimal threshold using ROC curves on `Annotated` set.
    3.  Diagnose patient based on % of patches exceeding the error threshold.

### **System 2: Supervised Classification with Attention (MIL)**
*Hypothesis:* A supervised model can learn to identify diagnostic features directly. Attention mechanisms allow the model to focus on relevant patches within a patient "bag".

* **Pipeline:**
    1.  **Feature Extraction:** Converting images to vectors. Compared 3 methods:
        * **AutoEncoder (Encoder):** Reusing System 1's encoder.
        * **ResNet50:** Pre-trained on ImageNet (captures texture/shapes).
        * **Triplet Loss:** Contrastive learning to enforce separation between healthy/infected patches in latent space.
    2.  **MIL Classification:**
        * **Gated Attention Mechanism:** Assigns a learnable weight to every patch (High weight = Suspicious).
        * **Aggregation:** Weighted average of patch vectors.
        * **Classifier:** Final binary decision (Healthy vs. Sick).

---

## 4. Experimental Results

### **System 1 Results**
* **Best Configuration:** **AutoEncoder with P99 Metric**.
* **Cross-Validation:** Achieved **AUC 0.953** and **0 False Negatives**, significantly outperforming MSE-based approaches.
* **HoldOut (Generalization):** AUC dropped to **0.87**. While Specificity remained high (89%), Sensitivity dropped (70%), indicating the fixed error threshold is sensitive to stain variations in new data.

<p align="center">
 <img width="45%" alt="System 1 CV Results" src="https://github.com/user-attachments/assets/1d98c230-96e7-4d3c-a4c8-cd335e6ce822" />
<img width="45%" alt="System 1 Holdout Results" src="https://github.com/user-attachments/assets/0fcbcabf-65a2-4786-8509-077710958b7b" />
</p>

### **System 2 Results**
* **Best Configuration:** **ResNet50 Features + Gated Attention**.
* **Cross-Validation:** **AUC ~0.99**. Perfect classification in validation folds.
* **HoldOut (Generalization):**
    * **ResNet50:** **AUC 0.905**. Balanced Sensitivity (67%) and Specificity (98%). Most robust clinical model.
    * **Triplet Loss:** Lower performance (AUC 0.827) due to overfitting on the training distribution.
    * **AutoEncoder Features:** Poor performance (AUC ~0.50), confirming reconstruction features are less discriminative for this task.

<p align="center">
   <img width="45%" alt="System 2 CV Results ResNet" src="https://github.com/user-attachments/assets/3630e9a8-4b2c-4b04-af0d-870c20dbb4c0" />

  <img src="path/to/system2_holdout_roc.png" alt="System 2 Holdout Results" width="45%">
   
</p>

---

## 5. Interpretability (Attention Maps)

For System 2, we validated the "Black Box" by visualizing the attention weights.
* **High Attention Patches:** Corresponded to areas with high bacterial load (visual confirmation).
* **Low Attention Patches:** Corresponded to healthy tissue or background.

This confirms the model is making decisions based on relevant biological signals, not noise.

---

## 6. Conclusions & Future Work

* **System 1 (AE P99):** Excellent **screening tool**. High specificity allows rapid exclusion of healthy patients. Requires threshold calibration for new scanners/stains.
* **System 2 (ResNet MIL):** Best **diagnostic tool**. Robust and reliable for final decision-making.
* **Future Improvements:**
    * Data augmentation to simulate stain variations (color normalization).
    * Finer-grained attention mechanisms to capture the <1% area occupied by bacteria.

---

## Code Structure

### **System 1 (Anomaly Detection)**
* `ae_models.py`: Definitions of AutoEncoder and VAE architectures.
* `train_ae_negativa.py`: Training script (Healthy patches only).
* `evaluate_anomaly.py`: Patch-level error calculation and reconstruction visualization.
* `patient_diagnosis_ae.py`: Patient-level aggregation logic.

### **System 2 (MIL + Attention)**
* **Feature Extraction:**
    * `S2_feature_extraction_with_resNET50.py`: Extracts 2048-dim vectors using pre-trained ResNet.
    * `S2_feature_extraction_with_triplet.py`: Extracts 128-dim vectors using Contrastive Learning model.
* **Training & Representation Learning:**
    * `S2_train_triplet.py`: Trains the projection head using Triplet Loss.
    * `S2_train_with_logs.py`: Main training loop for the MIL Classifier (5-Fold CV).
* **Evaluation:**
    * `S2_predict_holdout_patients.py`: Final HoldOut evaluation (Ensemble predictions, ROC with Std Dev, Confusion Matrix).
    * `S2_visualize_attention_unified.py`: Generates attention heatmaps (High vs Low attention patches).
* **Models:**
    * `S2_models.py`: Contains `NeuralNetwork_withAttention` and `GatedAttention` classes.

---

### Usage

To replicate the best performing model (System 2 ResNet):

1.  **Extract Features:**
    ```bash
    python S2_feature_extraction_with_resNET50.py
    ```
2.  **Train MIL Model:**
    ```bash
    python S2_train_with_logs.py --model ResNet
    ```
3.  **Evaluate on HoldOut:**
    ```bash
    python S2_predict_holdout_patients.py
    ```
