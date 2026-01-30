import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from dataset import get_dataloaders
from model import DeepSet
from train import train_one_epoch, validate

def run_experiment():
    device = torch.device("cpu")
    
    latent_dims = [2, 4, 8, 16, 32, 64, 128] 
    num_points = 100  # M 
    epochs = 20        
    batch_size = 64
    
    results = []

    train_loader, test_loader = get_dataloaders(batch_size, num_points)
    criterion = nn.CrossEntropyLoss()
    
    for dim in latent_dims:
        model = DeepSet(input_dim=2, latent_dim=dim, output_dim=10).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        best_acc = 0
        for epoch in range(epochs):
            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, val_acc = validate(model, test_loader, criterion, device)
            
            if val_acc > best_acc:
                best_acc = val_acc
                
            print(f"Epoch {epoch+1}: Val Acc: {val_acc:.2f}%")
        
        results.append(best_acc)

    plt.figure(figsize=(10, 6))
    plt.plot(latent_dims, results, marker='o', linestyle='-', color='b')
    plt.xscale('log', base=2) 
    plt.xlabel('Dimensión Latente (N)')
    plt.ylabel('Test Accuracy (%)')
    plt.title(f'Impacto de la Dimensión Latente en Deep Sets (M={num_points})')
    plt.grid(True, which="both", ls="-")
    plt.axvline(x=num_points, color='r', linestyle='--', label='N = M (Teórico Wagstaff)')
    plt.legend()
    plt.savefig('wagstaff_experiment_result.png')
    print("\nGráfica guardada como 'wagstaff_experiment_result.png'")

if __name__ == "__main__":
    run_experiment()