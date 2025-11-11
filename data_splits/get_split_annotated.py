#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os
from collections import defaultdict
import random

# --- CONFIG ---
ANNOT_XLSX = "/export/fhome/maed/HelicoDataSet/HP_WSI-CoordAllAnnotatedPatches.xlsx"
ANNOTATED_BASE = "/export/fhome/maed/HelicoDataSet/CrossValidation/Annotated"
OUT_DIR = "."  # carpeta donde guardar los txt/csv de salida
RANDOM_SEED = 42
TARGET_RATIO = 0.5  # ratio para threshold_set (0.5 => 50% / 50%)

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# --- CARGA ---
df = pd.read_excel(ANNOT_XLSX, dtype={"Pat_ID": str, "Section_ID": str, "Window_ID": str})
df = df.rename(columns=lambda s: s.strip())

# Asegurar que Window_ID mantiene ceros a la izquierda (5 dígitos)
df["Window_ID"] = df["Window_ID"].astype(str).str.zfill(5)

# Filtramos solo etiquetas válidas (1 y -1)
df = df[df["Presence"].isin([1, -1])].copy()

# Creamos identificador de carpeta: PatID_SectionID
df["Pat_Section"] = df["Pat_ID"].astype(str) + "_" + df["Section_ID"].astype(str)

# Contamos por carpeta cuántos positivos y negativos hay
agg = df.groupby("Pat_Section")["Presence"].value_counts().unstack(fill_value=0).rename(columns={1: "pos", -1: "neg"})
agg = agg.reset_index().fillna(0)
if "pos" not in agg.columns:
    agg["pos"] = 0
if "neg" not in agg.columns:
    agg["neg"] = 0

# lista de carpetas
folders = agg["Pat_Section"].tolist()
# guardamos diccionario de patches por carpeta
patches_by_folder = {k: df[df["Pat_Section"] == k].copy() for k in folders}

# --- GREEDY ASSIGNMENT (balancear positivos) ---
agg_sorted = agg.sort_values(by="pos", ascending=False).reset_index(drop=True)

threshold_folders = set()
validation_folders = set()

pos_count_thresh = 0
pos_count_val = 0
total_count_thresh = 0
total_count_val = 0

for _, row in agg_sorted.iterrows():
    folder = row["Pat_Section"]
    pos = int(row.get("pos", 0))
    neg = int(row.get("neg", 0))
    if pos_count_thresh <= pos_count_val:
        threshold_folders.add(folder)
        pos_count_thresh += pos
        total_count_thresh += pos + neg
    else:
        validation_folders.add(folder)
        pos_count_val += pos
        total_count_val += pos + neg

# Si alguna carpeta quedó sin asignar
all_assigned = threshold_folders.union(validation_folders)
for f in folders:
    if f not in all_assigned:
        if total_count_thresh <= total_count_val:
            threshold_folders.add(f)
            total_count_thresh += len(patches_by_folder[f])
            pos_count_thresh += (patches_by_folder[f]["Presence"]==1).sum()
        else:
            validation_folders.add(f)
            total_count_val += len(patches_by_folder[f])
            pos_count_val += (patches_by_folder[f]["Presence"]==1).sum()

# --- Construir dataframes finales ---
df_thresh = pd.concat([patches_by_folder[f] for f in sorted(threshold_folders)], ignore_index=True)
df_val = pd.concat([patches_by_folder[f] for f in sorted(validation_folders)], ignore_index=True)

# --- Balancear positivos/negativos ---
def balance_df(df_in, seed=RANDOM_SEED):
    pos_df = df_in[df_in["Presence"] == 1].copy()
    neg_df = df_in[df_in["Presence"] == -1].copy()
    n_pos = len(pos_df)
    n_neg = len(neg_df)
    if n_pos == 0:
        return df_in
    if n_neg > n_pos:
        neg_df_sample = neg_df.sample(n=n_pos, random_state=seed)
    else:
        neg_df_sample = neg_df
    df_bal = pd.concat([pos_df, neg_df_sample], ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)
    return df_bal

df_thresh_bal = balance_df(df_thresh)
df_val_bal = balance_df(df_val)

# --- GUARDAR RESULTADOS ---
os.makedirs(OUT_DIR, exist_ok=True)

def save_list_file(df_in, name):
    csv_path = os.path.join(OUT_DIR, f"{name}.csv")
    df_in.to_csv(csv_path, index=False)
    txt_path = os.path.join(OUT_DIR, f"{name}.txt")
    with open(txt_path, "w") as f:
        for _, r in df_in.iterrows():
            patsec = r["Pat_Section"]
            window = r["Window_ID"]
            f.write(f"{patsec}/{window}\n")
    return csv_path, txt_path

csv_t, txt_t = save_list_file(df_thresh_bal, "threshold_set_balanced")
csv_v, txt_v = save_list_file(df_val_bal, "validation_set_balanced")

# Guardar versiones no balanceadas
save_list_file(df_thresh, "threshold_set_unbalanced")
save_list_file(df_val, "validation_set_unbalanced")

# --- RESUMEN ---
print("=== RESUMEN ===")
print(f"Total annotated patches: {len(df)}")
print(f"Threshold set  (unbalanced): {len(df_thresh)}, positives: {(df_thresh['Presence']==1).sum()}, negatives: {(df_thresh['Presence']==-1).sum()}")
print(f"Threshold set  (balanced): {len(df_thresh_bal)}, positives: {(df_thresh_bal['Presence']==1).sum()}, negatives: {(df_thresh_bal['Presence']==-1).sum()}")
print(f"Validation set (unbalanced): {len(df_val)}, positives: {(df_val['Presence']==1).sum()}, negatives: {(df_val['Presence']==-1).sum()}")
print(f"Validation set (balanced): {len(df_val_bal)}, positives: {(df_val_bal['Presence']==1).sum()}, negatives: {(df_val_bal['Presence']==-1).sum()}")

print("\nFicheros generados:")
print(f" - {csv_t}")
print(f" - {txt_t}")
print(f" - {csv_v}")
print(f" - {txt_v}")
