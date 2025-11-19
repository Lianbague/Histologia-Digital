# Histologia-Digital

# SISTEMA 2 OVERVIEW
Arquitectura General del Sistema 2
1. Entrada: Todos los parches de un paciente (una "bolsa" o bag de imágenes).
2. Feature Extractor (Congelado): Transformamos cada imagen (256x256x3) en un vector numérico (embedding). Tus notas sugieren usar ResNet, ViT, o tu propio Autoencoder.
    - Estrategia: Usaremos el Encoder de tu Autoencoder ya entrenado como punto de partida (es la opción "espacio latente del autoencoder" de tus notas)
3. Attention Mechanism: Una red neuronal pequeña que mira los vectores y aprende pesos ($w_i$). Si un parche parece tejido canceroso, tendrá un peso alto. Si es tejido sano, peso bajo.
4. Clasificador: Promedio ponderado de los vectores $\rightarrow$ Clasificador Binario $\rightarrow$ Diagnóstico Paciente.