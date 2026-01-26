# prepare_demo.py
import torch
import torch.nn as nn
import torch.optim as optim
import os
from dataset import get_dataloaders
from model import DeepSet
from train import train_one_epoch

def save_models():
    if not os.path.exists('saved_models'):
        os.makedirs('saved_models')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Elegimos 3 niveles representativos para la demo
    latent_dims = [2, 16, 128] 
    epochs = 20 # Suficiente para MNIST
    
    train_loader, _ = get_dataloaders(batch_size=64, num_points=100)
    criterion = nn.CrossEntropyLoss()
    
    print("--- Entrenando modelos para la Demo ---")
    
    for dim in latent_dims:
        print(f"\nEntrenando Modelo con Latent Dim = {dim}...")
        model = DeepSet(latent_dim=dim).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        for epoch in range(epochs):
            loss, acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
            print(f"  Epoch {epoch+1}/{epochs} - Acc: {acc:.1f}%")
            
        # Guardar pesos
        save_path = f"saved_models/deepset_dim_{dim}.pth"
        torch.save(model.state_dict(), save_path)
        print(f"-> Guardado en {save_path}")

if __name__ == "__main__":
    save_models()