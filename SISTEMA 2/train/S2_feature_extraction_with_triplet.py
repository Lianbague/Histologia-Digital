import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import glob
from tqdm import tqdm

from S2_MIL.S2_train_triplet import ResNetEmbedding 

MODEL_PATH = 'S2_triplet_model_final.pth'
RAW_DATA_ROOT = '/export/fhome/maed/HelicoDataSet/CrossValidation/Cropped'
OUTPUT_DIR = '/export/fhome/maed03/Features_Triplet128'  
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ImageBagDataset(Dataset):
    def __init__(self, patch_paths, transform=None):
        self.patch_paths = patch_paths
        self.transform = transform
    def __len__(self): return len(self.patch_paths)
    def __getitem__(self, idx):
        try:
            img = Image.open(self.patch_paths[idx]).convert('RGB')
            if self.transform: img = self.transform(img)
            return img
        except:
            return None

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if not batch: return None
    return torch.utils.data.dataloader.default_collate(batch)

# PROCES D'EXTRACCIO
def generate_triplet_features():    
    print("Carregant model Contrastive Learning...")
    model = ResNetEmbedding(embedding_dim=128).to(DEVICE)
    
    # Carreguem els pesos entrenats
    # Assegurat que aquest fitxer existeix al directori on executes el script
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load('S2_triplet_model_final.pth', map_location=DEVICE))
        print("Pesos carregats correctament.")
    else:
        print("ERROR: No trobo 'S2_triplet_model_final.pth'.")
        return

    model.eval()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Transformacio
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    patient_ids = set([os.path.basename(f).split('_')[0] for f in glob.glob(os.path.join(RAW_DATA_ROOT, '*'))])
    print(f"Processant {len(patient_ids)} pacients cap a vectors de 128 dim...")

    with torch.no_grad():
        for pat_id in tqdm(patient_ids):
            save_path = os.path.join(OUTPUT_DIR, f"{pat_id}.pt")
            if os.path.exists(save_path): continue

            pat_sections = glob.glob(os.path.join(RAW_DATA_ROOT, f"{pat_id}_*"))
            all_patches = []
            for sec in pat_sections:
                all_patches.extend(glob.glob(os.path.join(sec, '*.png')))
            
            if not all_patches: continue

            dataset = ImageBagDataset(all_patches, transform=transform)
            loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4, collate_fn=collate_fn)

            patient_features = []
            for imgs in loader:
                if imgs is None: continue
                imgs = imgs.to(DEVICE)
                
                # Extraccio usant el model entrenat amb Triplet Loss
                feats = model(imgs) 
                
                patient_features.append(feats.cpu())
            
            if patient_features:
                patient_tensor = torch.cat(patient_features, dim=0)
                torch.save(patient_tensor, save_path)
    
    print("Proces finalitzat! Dades guardades a:", OUTPUT_DIR)

if __name__ == '__main__':
    generate_triplet_features()
