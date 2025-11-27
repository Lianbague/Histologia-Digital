# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.stats import ttest_rel, wilcoxon

# =============================================================================
# 1. RESULTATS DELS EXPERIMENTS (Dades dels 10 Folds)
# =============================================================================
# Valors d'Accuracy (o AUC) que hem obtingut:

results_ae_p99 = [1.00, 1.00, 0.8125, 0.8125, 1.00, 1.00, 0.8667, 1.00, 0.9231, 0.9333]
results_vae_p99 = [0.9286, 1.00, 0.8125, 0.8750, 1.00, 1.00, 0.9333, 0.7333, 0.8462, 0.8000]
results_ae_mse = [0.8571, 0.9333, 0.8750, 0.6875, 0.8571, 0.9286, 0.7333, 0.9333, 0.8462, 0.800]

# Posa'ls en un diccionari amb el nom que vulguis que surti a la gràfica
data_dict = {
    'AE (P99)': results_ae_p99,
    'VAE (P99)': results_vae_p99,
    'AE (MSE)': results_ae_mse
}

METRIC_NAME = "Accuracy" # 'Accuracy' O 'AUC'

# =============================================================================
# 2. GENERACIÓ DEL DATAFRAME I GRÀFICA
# =============================================================================

def main():
    # Convertim a format Pandas per a Seaborn
    data_list = []
    for model_name, values in data_dict.items():
        for val in values:
            data_list.append({'Model': model_name, 'Score': val})
    
    df = pd.DataFrame(data_list)
    
    # Configuració de la gràfica
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    # 1. Dibuixar el BOXPLOT
    # La caixa mostra els quartils (25% - 75%) i la línia del mig és la Mediana
    ax = sns.boxplot(x='Model', y='Score', data=df, 
                     width=0.5, palette="pastel", showfliers=False)
    
    # 2. Dibuixar els PUNTS INDIVIDUALS (Swarmplot o Stripplot)
    sns.stripplot(x='Model', y='Score', data=df, 
                  color='black', size=6, alpha=0.7, jitter=True)
    
    # Títols i etiquetes
    plt.title(f'Comparativa de Models: {METRIC_NAME} (10-Fold CV)', fontsize=14, fontweight='bold')
    plt.ylabel(METRIC_NAME, fontsize=12)
    plt.xlabel('Model Configuration', fontsize=12)
    
    # Afegir mitjana escrita
    means = df.groupby('Model')['Score'].mean()
    for i, model in enumerate(data_dict.keys()):
        plt.text(i, means[model], f"Mean: {means[model]:.3f}", 
                 horizontalalignment='center', size='small', color='darkred', weight='bold')

    plt.tight_layout()
    plt.savefig('comparativa_boxplots.png', dpi=300)
    print("Gràfica guardada com 'comparativa_boxplots.png'")
    plt.show()

    # =============================================================================
    # 3. TEST ESTADÍSTIC
    # =============================================================================
    print("\n--- Anàlisi Estadístic (T-Test Relacionat) ---")
    print("Si p-value < 0.05, la diferència és significativa.")
    
    models = list(data_dict.keys())
    # Comparar el primer amb la resta
    base_model = models[0] # AE (P99)
    
    for i in range(1, len(models)):
        compare_model = models[i]
        # Fem t-test paired perquè són els mateixos folds (mateixos pacients)
        stat, p_val = ttest_rel(data_dict[base_model], data_dict[compare_model])
        
        significatiu = "SÍ" if p_val < 0.05 else "NO"
        print(f"{base_model} vs {compare_model}: P-value = {p_val:.4f} -> Significatiu? {significatiu}")

if __name__ == "__main__":
    main()