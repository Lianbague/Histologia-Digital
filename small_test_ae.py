

# -*- coding: utf-8 -*-
import torch
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import os
import sys
import matplotlib.pyplot as plt
import numpy as np

from ae_models import AutoEncoderCNN, AEConfigs

# FUNCIONS D'AVALUACIÓ
def get_eval_transforms():
    """ Utilitza les mateixes transformacions que l'entrenament (SENSE NORMALITZACIO). """
    # --- CORRECCIÓ CLAU: ELIMINAR NORMALITZACIÓ EN AVALUACIÓ ---
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        # Eliminada: transforms.Normalize(...)
    ])

def calculate_reconstruction_error(image_path, model, device, show_reconstruction=False, save_dir="reconstructions"):
    if not os.path.exists(image_path):
        print(f"ERROR: No s'ha trobat la imatge a {image_path}")
        return None
        
    transforms_eval = get_eval_transforms()
    image = Image.open(image_path).convert('RGB')
    input_tensor = transforms_eval(image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        reconstruction = model(input_tensor)

    l_red = nn.MSELoss(reduction='none')(reconstruction, input_tensor).mean(dim=[1, 2, 3])
    error = l_red.item()

    if show_reconstruction:
        mean = torch.tensor([0.485, 0.456, 0.406]).to(device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).to(device).view(1, 3, 1, 1)

        # input_denorm = (input_tensor * std + mean).clamp(0, 1)
        # recon_denorm = (reconstruction * std + mean).clamp(0, 1)

        inp_img = input_denorm.squeeze().permute(1, 2, 0).cpu().numpy()
        recon_img = recon_denorm.squeeze().permute(1, 2, 0).cpu().numpy()

        os.makedirs(save_dir, exist_ok=True)
        fname = os.path.basename(image_path).replace('.png', '_reconstruction.png')

        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(inp_img)
        plt.title("Original")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(recon_img)
        plt.title("Reconstrucció")
        plt.axis('off')
        plt.suptitle(f"Reconstruction Error: {error:.6f}")

        save_path = os.path.join(save_dir, fname)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"Imatge de reconstrucció guardada a: {save_path}")

    return error


if __name__ == '__main__':

    #FOLDERS_DIR = "patches_small_test.csv"
    images = [['/export/fhome/maed/HelicoDataSet/CrossValidation/Annotated/B22-174_0/00172.png',-1],
    ['/export/fhome/maed/HelicoDataSet/CrossValidation/Annotated/B22-304_0/00318_Aug6.png', 1],
    ['/export/fhome/maed/HelicoDataSet/CrossValidation/Annotated/B22-174_0/00182.png',-1],
    ['/export/fhome/maed/HelicoDataSet/CrossValidation/Annotated/B22-68_0/00352.png',-1],
    ['/export/fhome/maed/HelicoDataSet/CrossValidation/Annotated/B22-116_0/00256_Aug8.png',1],
    ['/export/fhome/maed/HelicoDataSet/CrossValidation/Annotated/B22-167_0/00006.png',-1],
    ['/export/fhome/maed/HelicoDataSet/CrossValidation/Annotated/B22-302_0/00175_Aug8.png',1],
    ['/export/fhome/maed/HelicoDataSet/CrossValidation/Annotated/B22-174_0/00239_Aug4.png',1],
    ['/export/fhome/maed/HelicoDataSet/CrossValidation/Annotated/B22-304_0/00783_Aug1.png',1],
    ['/export/fhome/maed/HelicoDataSet/CrossValidation/Annotated/B22-304_0/01757_Aug7.png',1],
    ['/export/fhome/maed/HelicoDataSet/CrossValidation/Annotated/B22-187_0/00591.png',-1],
    ['/export/fhome/maed/HelicoDataSet/CrossValidation/Annotated/B22-50_0/01342.png',1],
    ['/export/fhome/maed/HelicoDataSet/CrossValidation/Cropped/B22-106_1/10.png', -1],
    ['/export/fhome/maed/HelicoDataSet/CrossValidation/Annotated/B22-101_0/00180.png', 1]
    ]



    
    MODEL_SAVE_PATH = "/fhome/maed03/autoencoder_negativa_best_L1Loss.pth" # Fitxer entrenat
    CONFIG = '1'
    # Per evitar l'error de cuda, utilitzem l'opcio de cpu si la GPU no esta disponible directament
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # --------------------------------------------------------------------------------
    
    if not os.path.exists(MODEL_SAVE_PATH):
        print(f"ERROR: No s'ha trobat el model {MODEL_SAVE_PATH}. Assegura't que l'entrenament ha finalitzat correctament.")
        sys.exit(1)
        
    # Carregar el model
    print(f"Carregant model entrenat a {DEVICE}...")
    config = AEConfigs(config_id='1', input_channels=3)
    model = AutoEncoderCNN(
    net_paramsEnc=config.net_paramsEnc, 
    inputmodule_paramsDec=config.inputmodule_paramsDec, 
    net_paramsDec=config.net_paramsDec
    )

    # Carregar els pesos
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
    model.to(DEVICE)
    
    print("\n--- RESULTATS DE L'AVALUACIO ---")

    for image_path, presence in images:
        print(f"\nAvaluant la imatge: {image_path}")
        error = calculate_reconstruction_error(image_path, model, DEVICE, show_reconstruction=False)
        print(f"Error de reconstruccio: {error:.6f}, la imatge era:", "Anomalia" if presence == 1 else "Sana")