import torch  
import torch.nn as nn  

class Encoder(nn.Module):
    """
    Classe que implementa el codificador (encoder) de l'autoencoder.
    S'encarrega de comprimir la imatge d'entrada en una representació latent de menor dimensió.
    """
    def __init__(self, net_params):
        super(Encoder, self).__init__()
        layers = []
        in_channels = net_params['input_channels']  # Nombre de canals de la imatge d'entrada
        
        # Construcció de les capes convolucionals del codificador
        for out_channels, kernel_size, stride, padding in net_params['conv_layers']: # conv_layers: Llista de tuples (out_channels, kernel_size, stride, padding)
            # Afegeix una capa convolucional amb els paràmetres especificats
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))
            # Afegeix una funció d'activació ReLU després de cada convolució
            layers.append(nn.ReLU())
            in_channels = out_channels  # Actualitza el nombre de canals per la següent capa
        
        # Crea una seqüència de capes que s'executaran en ordre
        self.encoder = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Realitza la codificació de la imatge d'entrada.
        """
        return self.encoder(x)
    
class Decoder(nn.Module):
    """
    Classe que implementa el descodificador (decoder) de l'autoencoder.
    S'encarrega de reconstruir la imatge original a partir de la representació latent.
    """
    def __init__(self, net_params):
        super(Decoder, self).__init__()
        layers = []
        in_channels = net_params['input_channels']
        
        # Construcció de les capes deconvolucionals del descodificador
        for out_channels, kernel_size, stride, padding in net_params['conv_layers']:
            # Afegeix una capa de convolució transposada (deconvolució)
            layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding))
            # Afegeix una funció d'activació ReLU després de cada deconvolució
            layers.append(nn.ReLU())
            in_channels = out_channels  # Actualitza el nombre de canals per la següent capa
        
        # Crea una seqüència de capes que s'executaran en ordre
        self.decoder = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Realitza la descodificació de la representació latent.
        """
        return self.decoder(x)
    
class AutoEncoderCNN(nn.Module):
    """
    Classe principal que implementa l'autoencoder complet.
    Combina el codificador i el descodificador en una única xarxa neuronal.
    """
    def __init__(self, enc_params, dec_params):
        super(AutoEncoderCNN, self).__init__()
        self.encoder = Encoder(enc_params)  
        self.decoder = Decoder(dec_params) 
    
    def forward(self, x):
        encoded = self.encoder(x)     # Codifica la imatge a un espai latent
        decoded = self.decoder(encoded) # Descodifica per reconstruir la imatge
        return decoded