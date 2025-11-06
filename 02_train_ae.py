import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pickle
import os
from tqdm import tqdm

# Importa les definicions de model
from ae_model import AutoEncoderCNN, AEConfigs # ASSUMPCIÓ: El codi del model està a ae_model.py

# --- CONFIGURACIÓ DE RUTES I HIPERPARÀMETRES ---
INPUT_FILE = 'sane_patches_data.pkl' # Fitxer de dades precarregades
MODEL_OUTPUT_FOLDER = 'autoencoders/models'
MODEL_NAME = 'sane_patches_ae_model.pth'

NUM_EPOCHS = 50 
BATCH_SIZE = 64
LEARNING_RATE = 1e-4

# --- 1. CONFIGURACIÓ DE GPU (Segons clúster/Slurm) ---
# Utilitzem DDP (Distributed Data Parallel) si vols que Slurm assigni diverses GPUs, 
# però per començar amb una:

# Assegurem que PyTorch utilitzi la GPU assignada per Slurm.
# Slurm normalment exporta una variable CUDA_VISIBLE_DEVICES.
if torch.cuda.is_available():
    device = torch.device("cuda")
    # Si estem en un entorn DDP, caldria establir el rank (però simplificarem a 1 GPU)
    print("GPU detectada.")
else:
    # Aquesta línia hauria de ser rara si Slurm t'ha donat una GPU.
    device = torch.device("cpu")
print(f"Utilitzant device: {device}")


# --- 2. CÀRREGA DE DADES PRECÀRREGADES ---
print(f"2. Carregant dades des de {INPUT_FILE}...")
try:
    with open(INPUT_FILE, 'rb') as f:
        X_train_np = pickle.load(f)
    print("Dades carregades amb èxit.")
except FileNotFoundError:
    print(f"ERROR: No s'ha trobat el fitxer de dades precarregades: {INPUT_FILE}.")
    exit()

# Converteix a PyTorch Tensor
X_train_tensor = torch.from_numpy(X_train_np)

# Utilitzem TensorDataset per evitar el __getitem__ manualment
# Input = Target per a Autoencoders
train_dataset = TensorDataset(X_train_tensor, X_train_tensor) 

train_loader = DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    num_workers=4 # Pots augmentar per velocitat si la teva CPU ho permet
)

# --- 3. INICIALITZACIÓ DEL MODEL ---
Config = '1'
net_paramsEnc, net_paramsDec = AEConfigs(Config)
model = AutoEncoderCNN(net_paramsEnc, net_paramsDec)
model.to(device)

# Funció de pèrdua (Loss): Mean Squared Error (L2 Loss)
criterion = nn.MSELoss() 
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --- 4. BUCLE PRINCIPAL D'ENTRENAMENT ---
print("\nIniciant l'entrenament de l'Autoencoder...")
for epoch in range(NUM_EPOCHS):
    model.train() 
    running_loss = 0.0
    
    for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
        
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(train_dataset)
    print(f"\n[Epoch {epoch+1}/{NUM_EPOCHS}] Loss: {epoch_loss:.6f}")

# --- 5. DESA EL MODEL ---
os.makedirs(MODEL_OUTPUT_FOLDER, exist_ok=True)
model_path = os.path.join(MODEL_OUTPUT_FOLDER, MODEL_NAME)
torch.save(model.state_dict(), model_path)
print(f"\nEntrenament finalitzat. Model desat a: {model_path}")