#!/bin/bash
#SBATCH -n 4 
#SBATCH -N 1 
#SBATCH -D /fhome/maed03
#SBATCH -t 4-00:05
#SBATCH -p tfg 
#SBATCH --mem 12288 
#SBATCH -o %x_%u_%j.out
#SBATCH -e %x_%u_%j.err
#SBATCH --gres gpu:1 

sleep 3


# Activa l'entorn virtual de Python
source /fhome/maed03/MyVirtualEnv/bin/activate

# Envia l'scrip a executar a la cua
python /fhome/maed03/train_ae_negativa.py

echo "Proces finalitzat."