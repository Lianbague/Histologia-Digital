import torch
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import os
import sys

from ae_models import AutoEncoderCNN, AEConfigs

# FUNCIONS D'AVALUACIÓ
def get_eval_transforms():
    """ Utilitza les mateixes transformacions i normalitzacio que l'entrenament. """
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def calculate_reconstruction_error(image_path, model, device):
    """ Carrega una patch i calcula el seu error de reconstruccio. """
    if not os.path.exists(image_path):
        print(f"ERROR: No s'ha trobat la imatge a {image_path}")
        return None
        
    transforms = get_eval_transforms()
    
    # Carregar la imatge
    image = Image.open(image_path).convert('RGB')
    input_tensor = transforms(image).unsqueeze(0).to(device) 

    # Reconstruccio
    model.eval()
    with torch.no_grad():
        reconstruction = model(input_tensor)
        
    # Calcular l'Error de Reconstruccio (MSE / L2 Loss)
    l_red = nn.MSELoss(reduction='none')(reconstruction, input_tensor).mean(dim=[1, 2, 3])
    
    return l_red.item()


if __name__ == '__main__':
    
    MODEL_SAVE_PATH = 'autoencoder_negativa_best.pth' # Fitxer entrenat
    CONFIG = '1'
    # Per evitar l'error de cuda, utilitzem l'opcio de cpu si la GPU no esta disponible directament
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    #* Modificar Rutes
    # NEGATIVA (Sana): Dins Cropped, d'un PatID NEGATIVA (Ex: B22-03)
    PATH_PATCH_NEGATIVA = '/export/fhome/maed/HelicoDataSet/CrossValidation/Cropped/B22-106_1/10.png' 
    # POSITIVA (Anomalia): Dins Annotated, d'un PatID ALTA/BAIXA (Ex: B22-19)
    PATH_PATCH_POSITIVA = '/export/fhome/maed/HelicoDataSet/CrossValidation/Annotated/B22-101_0/00180.png'
    # --------------------------------------------------------------------------------
    
    if not os.path.exists(MODEL_SAVE_PATH):
        print(f"ERROR: No s'ha trobat el model {MODEL_SAVE_PATH}. Assegura't que l'entrenament ha finalitzat correctament.")
        sys.exit(1)
        
    # Carregar el model
    print(f"Carregant model entrenat a {DEVICE}...")
    net_paramsEnc, net_paramsDec, inputmodule_paramsDec = AEConfigs(CONFIG)
    model = AutoEncoderCNN(inputmodule_paramsDec, net_paramsEnc, net_paramsDec)
    
    # Carregar els pesos
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
    model.to(DEVICE)
    
    print("\n--- RESULTATS DE L'AVALUACIO ---")
    
    # Test amb Patch NEGATIVA (Sana)
    error_neg = calculate_reconstruction_error(PATH_PATCH_NEGATIVA, model, DEVICE)
    
    # Test amb Patch POSITIVA (Anomalia)
    error_pos = calculate_reconstruction_error(PATH_PATCH_POSITIVA, model, DEVICE)
    
    if error_neg is not None and error_pos is not None:
        print(f"Error Reconstruccio (NEGATIVA - Sana): {error_neg:.6f}")
        print(f"Error Reconstruccio (POSITIVA - Anomalia): {error_pos:.6f}")
        
        if error_pos > error_neg:
            ratio = error_pos / error_neg
            print(f"\n Deteccio d'Anomalia amb exit: L_red_POS es {ratio:.2f}x mes gran que L_red_NEG.")
        else:
            print("\n Advertencia: L'Error Positiu no es major que el Negatiu. El model podria no ser discriminatori.")
