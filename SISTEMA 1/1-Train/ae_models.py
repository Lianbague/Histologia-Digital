# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

class AEConfigs:
    def __init__(self, config_id='1', input_channels=3):
        """Classe per emmagatzemar les configuracions de l'AutoEncoder CNN."""
        self.net_paramsEnc = {}
        self.net_paramsDec = {}
        self.inputmodule_paramsEnc = {}
        self.inputmodule_paramsDec = {}

        if config_id == '1':
            # ENCODER: 256 -> 128 -> 64 -> 32 -> 16 -> 8
            self.net_paramsEnc['block_configs'] = [
                [32, 3, 2, 1],  
                [64, 3, 2, 1], 
                [128, 3, 2, 1], 
                [256, 3, 2, 1], 
                [512, 3, 2, 1]   
            ]
            
            # DECODER: 8 -> 16 -> 32 -> 64 -> 128 -> 256
            # (El 5e element es l'output_padding)
            self.net_paramsDec['block_configs'] = [
                [256, 3, 2, 1, 1], 
                [128, 3, 2, 1, 1], 
                [64, 3, 2, 1, 1],  
                [32, 3, 2, 1, 1],  
                [3, 3, 2, 1, 1] 
            ]
            
            self.inputmodule_paramsEnc['num_input_channels'] = input_channels
            self.inputmodule_paramsDec['num_input_channels'] = self.net_paramsEnc['block_configs'][-1][0] # 512
            
        else:
            raise ValueError(f"Config ID '{config_id}' no reconegut.")
    
class AutoEncoderCNN(nn.Module):
    def __init__(self, net_paramsEnc, inputmodule_paramsDec, net_paramsDec):
        super(AutoEncoderCNN, self).__init__()

        # ENCODER
        layers = []
        in_channels = 3 # RGB input
        for block in net_paramsEnc['block_configs']:
            layers.append(ConvBlock(in_channels, block[0], block[1], block[2], block[3]))
            in_channels = block[0]
        self.encoder = nn.Sequential(*layers)

        # DECODER
        layers = []
        in_channels = inputmodule_paramsDec['num_input_channels'] 
        for i, block in enumerate(net_paramsDec['block_configs']):
            is_last = (i == len(net_paramsDec['block_configs']) - 1)
            output_p = block[4] if len(block) > 4 else 0 
            
            if is_last:
                # Per corregir l'error de dimensio 226 vs 256, forcem output_padding=1
                final_output_padding = 1 
                
                tconv_final = nn.ConvTranspose2d(in_channels, block[0], kernel_size=block[1], 
                                                 stride=block[2], padding=block[3], 
                                                 output_padding=final_output_padding)
                
                layers.append(nn.Sequential(tconv_final, nn.Sigmoid()))
            else: 
                layers.append(TConvBlock(in_channels, block[0], block[1], block[2], block[3], output_padding=output_p))
                in_channels = block[0]
        
        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        encoded = self.encoder(x)
        reconstructed = self.decoder(encoded)
        return reconstructed


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class TConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding=0):
        super(TConvBlock, self).__init__()
        self.tconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, 
                                        stride=stride, padding=padding, output_padding=output_padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.tconv(x)))
        
# VAE --> NOU MODEL        
class VariationalAutoEncoderCNN(nn.Module):
    def __init__(self, inputmodule_paramsEnc, net_paramsEnc, inputmodule_paramsDec, net_paramsDec, latent_dim=128, img_size=128):
        super(VariationalAutoEncoderCNN, self).__init__()

        # --- ENCODER ---
        layers = []
        in_channels = inputmodule_paramsEnc['num_input_channels']
        for block in net_paramsEnc['block_configs']:
            layers.append(ConvBlock(in_channels, block[0], block[1], block[2], block[3]))
            in_channels = block[0]
        self.encoder_conv = nn.Sequential(*layers)

        # --- CÀLCUL AUTOMÀTIC DE MIDES (La part nova i important) ---
        # Creem una imatge falsa amb la mida que estem usant (ex: 128 o 256)
        # per veure com queda després de les convolucions.
        with torch.no_grad():
            dummy_input = torch.zeros(1, inputmodule_paramsEnc['num_input_channels'], img_size, img_size)
            dummy_output = self.encoder_conv(dummy_input)
            
            # Guardem la forma espacial (ex: 512, 4, 4) i la mida aplanada (ex: 8192)
            self.spatial_shape = dummy_output.shape[1:] 
            self.flattened_size = dummy_output.numel()
            
        print(f"Model VAE inicialitzat. Input: {img_size}x{img_size} -> Flattened: {self.flattened_size}")

        # Capes lineals dinàmiques
        self.fc_mu = nn.Linear(self.flattened_size, latent_dim)
        self.fc_log_var = nn.Linear(self.flattened_size, latent_dim)

        # --- DECODER ---
        self.decoder_input = nn.Linear(latent_dim, self.flattened_size)

        layers_dec = []
        in_channels = inputmodule_paramsDec['num_input_channels']
        for i, block in enumerate(net_paramsDec['block_configs']):
            is_last = (i == len(net_paramsDec['block_configs']) - 1)
            output_p = block[4] if len(block) > 4 else 0 
            
            if is_last:
                tconv_final = nn.ConvTranspose2d(in_channels, block[0], kernel_size=block[1], 
                                                 stride=block[2], padding=block[3], 
                                                 output_padding=output_p)
                layers_dec.append(nn.Sequential(tconv_final, nn.Sigmoid()))
            else: 
                layers_dec.append(TConvBlock(in_channels, block[0], block[1], block[2], block[3], output_padding=output_p))
                in_channels = block[0]
        
        self.decoder_tconv = nn.Sequential(*layers_dec)

    def encode(self, x):
        x = self.encoder_conv(x)
        x = torch.flatten(x, start_dim=1)
        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.decoder_input(z)
        
        # --- CORRECCIÓ CRÍTICA ---
        # En lloc de fer .view(-1, 512, 8, 8) hardcoded, 
        # usem la mida que hem calculat al __init__
        x = x.view(-1, *self.spatial_shape) 
        
        x = self.decoder_tconv(x)
        return x

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        reconstruction = self.decode(z)
        return reconstruction, mu, log_var