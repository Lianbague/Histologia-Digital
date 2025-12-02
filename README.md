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
The dataset (provided by QUIRON Salut) consists of high-resolution histological images of gastric biopsies divided into patches.

