import numpy as np
import torch
import torch.nn as nn

class CNN1D(nn.Module):
    """1D CNN for time series prediction"""
    def __init__(self, 
                 input_size: int, 
                 output_size: int, 
                 device: str = 'cpu', 
                 dropout_rate: float = 0.2):
        super(CNN1D, self).__init__()
        self.device = device
        self.input_size = input_size 
        self.output_size = output_size
        
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout_rate),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(), 
            nn.MaxPool1d(2),
            nn.Dropout(dropout_rate)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(64 * (input_size // 4), 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, output_size)
        )
        
        self.to(device)
        self._init_weights()
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        x = x.unsqueeze(1)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _init_weights(self) -> None:
        """Inicializa los pesos usando inicialización de Kaiming/He para capas convolucionales con ReLU"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'conv' in name:
                    # Inicialización He para capas convolucionales
                    nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
                else:
                    # Inicialización He para capas lineales
                    nn.init.kaiming_uniform_(param, mode='fan_in', nonlinearity='relu')
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
            'input_size': self.input_size,
            'output_size': self.output_size,
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
        self.input_size = checkpoint['input_size']
        self.output_size = checkpoint['output_size']
        self.device = checkpoint['device']
        self.to(self.device)

class CNNLSTM(nn.Module):
    """Combined CNN and LSTM for time series prediction"""
    def __init__(self, 
                 hidden_size: int, 
                 output_size: int, 
                 device: str = 'cpu', 
                 dropout: float = 0.2, 
                 num_layers: int = 2):
        super(CNNLSTM, self).__init__()
        self.device = device
        self.num_layers = num_layers
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1), 
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool1d(2)
        )
        self.lstm = nn.LSTM(64, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        self.to(device)
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        x = x.unsqueeze(1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        _, (h_n, _) = self.lstm(x)
        x = self.fc(h_n[-1])  # Using the last layer's hidden state
        return x
    
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
        self.device = checkpoint['device']
        self.to(self.device)