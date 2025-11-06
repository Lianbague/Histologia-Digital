# -*- coding: utf-8 -*-
# get_negativa_patients.py
import pandas as pd
import os
import sys 

# Ruta al PatientDiagnosis.csv
csv_path = '/export/fhome/maed/HelicoDataSet/PatientDiagnosis.csv'
output_file = 'negativa_patients.txt'

try:
    # Si el CSV conte caracters especials a les capcaleres, usem encoding
    # El fitxer original PatientDiagnosis.csv ja usa accents: CODI,DENSITAT
    df = pd.read_csv(csv_path, encoding='utf-8') 
    
    # Comprovem la carrega
    if df.empty:
        print("ADVERTENCIA: El CSV esta buit.")
        sys.exit(1)
    
    # Columna CODI es Pat_ID, Columna DENSITAT es la classificacio
    # ATENCIO: Utilitzem 'NEGATIVA' amb majuscules per a coincidencia exacta
    negativa_patients = df[df['DENSITAT'] == 'NEGATIVA']['CODI'].tolist()
    
    # Guardar els IDs en un fitxer de text
    with open(output_file, 'w') as f:
        for patient_id in negativa_patients:
            f.write(f"{patient_id}\n")
            
    print(f"Llista de pacients NEGATIVA guardada a: {output_file}")
    print(f"Total pacients NEGATIVA: {len(negativa_patients)}")

except Exception as e:
    # Utilitzem un string sense accents per garantir que l'error s'imprimeix
    print(f"ERROR inesperat: {e}")
    sys.exit(1)