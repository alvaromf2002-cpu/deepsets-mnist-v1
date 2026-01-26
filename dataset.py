# dataset.py
import torch
from torchvision import datasets, transforms
import numpy as np

class MNISTPointCloud(datasets.MNIST):
    def __init__(self, root, train=True, transform=None, num_points=100, download=False):
        super().__init__(root, train=train, transform=transform, download=download)
        self.num_points = num_points

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        
        # Convertir a numpy y normalizar coordenadas
        img_np = img.numpy()
        
        # Encontrar los píxeles con valor > 0 (la tinta del dígito)
        # coords será un array de (N, 2) con las coordenadas (y, x)
        coords = np.argwhere(img_np > 0).astype(np.float32)
        
        # Normalizar coordenadas entre 0 y 1 (dividiendo por 28x28)
        coords /= 28.0
        
        # Muestreo de puntos para tener un tamaño fijo M (num_points)
        if coords.shape[0] > 0:
            # Si hay puntos, muestreamos aleatoriamente
            choice_idx = np.random.choice(coords.shape[0], self.num_points, replace=True)
            point_set = coords[choice_idx, :]
        else:
            # Caso raro: imagen vacía (ruido), rellenar con ceros
            point_set = np.zeros((self.num_points, 2), dtype=np.float32)
            
        # Convertir a tensor
        point_set = torch.from_numpy(point_set)
        
        return point_set, target

def get_dataloaders(batch_size=64, num_points=100):
    train_dataset = MNISTPointCloud(root='./data', train=True, download=True, num_points=num_points)
    test_dataset = MNISTPointCloud(root='./data', train=False, download=True, num_points=num_points)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader