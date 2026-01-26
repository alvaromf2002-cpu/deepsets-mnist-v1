# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepSet(nn.Module):
    def __init__(self, input_dim=2, latent_dim=10, output_dim=10):
        super(DeepSet, self).__init__()
        self.latent_dim = latent_dim
        
        # --- Red Phi (Encoder) ---
        # Eleva cada punto 2D a una representación latente de mayor dimensión
        # Zaheer recomienda varias capas para Phi [cite: 76]
        self.phi = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim) # Proyectamos al espacio latente crítico
        )
        
        # --- Red Rho (Decoder/Classifier) ---
        # Procesa la suma agregada
        self.rho = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        # x shape: (Batch, Num_Points, 2)
        
        # 1. Aplicar Phi a cada punto individualmente
        # Compartimos pesos a través de todos los puntos (weight sharing)
        x_phi = self.phi(x) # (Batch, Num_Points, Latent_Dim)
        
        # 2. Agregación (Sum Pooling)
        # Esta operación garantiza la invarianza a permutaciones 
        x_agg = x_phi.sum(dim=1) # (Batch, Latent_Dim)
        
        # 3. Aplicar Rho al vector global
        output = self.rho(x_agg) # (Batch, Output_Dim/Classes)
        
        return output