#!/bin/bash
#SBATCH -A maed03
#SBATCH -p dcc 
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 03:00:00
#SBATCH -o train_ae_%j.out
#SBATCH -e train_ae_%j.err

echo "Iniciant proces d'entrenament de l'Autoencoder..."

# Carregar i activar l'entorn virtual
source /export/fhome/maed03/MyVirtualEnv/bin/activate

# Anar a la carpeta del projecte (Canvia a /export/fhome/maed03/challenge3 si el teu codi hi és)
cd /export/fhome/maed03/

# Executar l'script Python d'entrenament
python3 train_ae_negativa.py

echo "Proces finalitzat."
