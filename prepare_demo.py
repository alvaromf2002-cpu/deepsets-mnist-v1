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
    
    device = torch.device("cpu")
    # 3 latent spaces sizes for demo
    latent_dims = [2, 16, 128] 
    epochs = 20 
    
    train_loader, _ = get_dataloaders(batch_size=64, num_points=100)
    criterion = nn.CrossEntropyLoss()
    
    for dim in latent_dims:
        model = DeepSet(latent_dim=dim).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        for epoch in range(epochs):
            loss, acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
            print(f"  Epoch {epoch+1}/{epochs} - Acc: {acc:.1f}%")
            
        # Save weights on saved_models/ to use in demo
        save_path = f"saved_models/deepset_dim_{dim}.pth"
        torch.save(model.state_dict(), save_path)

if __name__ == "__main__":
    save_models()