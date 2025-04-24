import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Union, Dict
from sklearn.metrics import mean_squared_error, mean_absolute_error

class LSTM(nn.Module):
    """LSTM network for time series prediction"""
    def __init__(self, 
                 input_size: int, 
                 hidden_size: int, 
                 num_layers: int, 
                 output_size: int, 
                 dropout: float = 0.2, 
                 device: str = 'cpu'):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        self.to(device)
        self._init_weights()
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        # Ensure input has shape (batch_size, sequence_length, input_size)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # Add sequence dimension if missing
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

    def _init_weights(self) -> None:
        """Inicializa los pesos usando inicialización de Xavier/Glorot"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def save_model(self, path: str) -> None:
        """
        Guarda los pesos del modelo en la ruta especificada
        
        Args:
            path: Ruta donde guardar el modelo
        """
        torch.save({
            'model_state_dict': self.state_dict(),
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'device': self.device
        }, path)

    def load_model(self, path: str) -> None:
        """
        Carga los pesos del modelo desde la ruta especificada
        
        Args:
            path: Ruta desde donde cargar el modelo
        """
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.hidden_size = checkpoint['hidden_size']
        self.num_layers = checkpoint['num_layers'] 
        self.device = checkpoint['device']
        self.to(self.device)