import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepSet(nn.Module):
    def __init__(self, input_dim=2, latent_dim=10, output_dim=10):
        super(DeepSet, self).__init__()
        self.latent_dim = latent_dim
        
        # Encoder
        self.phi = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim) # Proyectamos al espacio latente crítico
        )
        
        # Recoder(Classifier) 
        self.rho = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        # x shape: (Batch, Num_Points, 2)
        x_phi = self.phi(x) # (Batch, Num_Points, Latent_Dim)
        x_agg = x_phi.sum(dim=1) # (Batch, Latent_Dim)
        output = self.rho(x_agg) # (Batch, Output_Dim/Classes)
        
        return output