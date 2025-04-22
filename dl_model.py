import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Union, Dict
from sklearn.metrics import mean_squared_error, mean_absolute_error

class WindowDataset(Dataset):
    """Dataset interno para manejar las ventanas de un conjunto específico"""
    def __init__(self, inputs: np.ndarray, labels: np.ndarray):
        self.inputs = torch.FloatTensor(inputs)
        self.labels = torch.FloatTensor(labels)
        
    def __len__(self) -> int:
        return len(self.inputs)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[idx], self.labels[idx]

class TimeSeriesDataset(Dataset):
    """Dataset class for time series data with flexible windowing"""
    
    def __init__(self, data: np.ndarray, input_width: int, label_width: int):
        """
        Inicializa el dataset de series temporales
        
        Args:
            data: Array de datos de la serie temporal
            input_width: Ancho de la ventana de entrada
            label_width: Ancho de la ventana de salida
        """
        self.data = data
        self.input_width = input_width
        self.label_width = label_width
        
        # Calcular tamaño total de la ventana
        self.total_window_size = input_width + label_width
        
        # Definir slices para inputs y labels
        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]
        
        self.label_start = self.total_window_size - label_width
        self.labels_slice = slice(self.label_start, self.label_start + label_width)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]
        
        # Calcular número total de ventanas
        self.n_windows = len(data) - self.total_window_size + 1
        
        # Inicializar conjuntos de datos
        self.train_data = None
        self.val_data = None
        self.test_data = None
        
        # Inicializar ventanas de datos
        self.train_windows = None
        self.val_windows = None
        self.test_windows = None
        
        # Inicializar dataloaders
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        
    def __len__(self) -> int:
        """Retorna el número total de ventanas disponibles"""
        return self.n_windows
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Obtiene una ventana de datos con sus etiquetas
        
        Args:
            idx: Índice de la ventana
            
        Returns:
            Tuple con (inputs, labels) como tensores de PyTorch
        """
        # Obtener la ventana completa
        window = self.data[idx:idx + self.total_window_size]
        
        # Separar inputs y labels
        inputs = window[self.input_slice]
        labels = window[self.labels_slice]
        
        return torch.FloatTensor(inputs), torch.FloatTensor(labels)
    
    def plot_window(self, idx: int = 0, figsize: Tuple[int, int] = (12, 6)) -> None:
        """
        Grafica una ventana específica del dataset
        
        Args:
            idx: Índice de la ventana a graficar
            figsize: Tamaño de la figura
        """
        inputs, labels = self[idx]
        
        plt.figure(figsize=figsize)
        
        # Graficar inputs
        plt.plot(self.input_indices, inputs.numpy(), 
                'b-', label='Inputs', marker='.', zorder=-10)
        
        # Graficar labels
        plt.plot(self.label_indices, labels.numpy(), 
                'g-', marker='.', zorder=-10)
        plt.scatter(self.label_indices, labels.numpy(),
                   edgecolors='k', marker='s', label='Labels',
                   c='green', s=64)
        
        plt.title(f'Ventana {idx} del Dataset')
        plt.xlabel('Tiempo')
        plt.ylabel('Valor')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # Ocultar spines superior y derecho
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    def get_batch(self, batch_size: int = 32, shuffle: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Obtiene un batch de datos
        
        Args:
            batch_size: Tamaño del batch
            shuffle: Si se debe mezclar los datos
            
        Returns:
            Tuple con (inputs, labels) como tensores de PyTorch
        """
        indices = np.arange(len(self))
        if shuffle:
            np.random.shuffle(indices)
            
        # Seleccionar índices para el batch
        batch_indices = indices[:batch_size]
        
        # Obtener datos para el batch
        batch_inputs = []
        batch_labels = []
        
        for idx in batch_indices:
            inputs, labels = self[idx]
            batch_inputs.append(inputs)
            batch_labels.append(labels)
            
        return torch.stack(batch_inputs), torch.stack(batch_labels)

    def split_data(self, train_pct: float = 0.7, val_pct: float = 0.15) -> None:
        """
        Divide los datos en conjuntos de train, validation y test según porcentajes
        
        Args:
            train_pct: Porcentaje para entrenamiento (default: 0.7)
            val_pct: Porcentaje para validación (default: 0.15)
            
        Nota:
            El porcentaje restante se usa para test
        """
        if train_pct + val_pct >= 1.0:
            raise ValueError("La suma de train_pct y val_pct debe ser menor a 1.0")
            
        n = len(self.data)
        train_end = int(n * train_pct)
        val_end = int(n * (train_pct + val_pct))
        
        self.train_data = self.data[:train_end]
        self.val_data = self.data[train_end:val_end]
        self.test_data = self.data[val_end:]
        
        print(f"\nDivisión de datos:")
        print(f"Train: {len(self.train_data)} muestras ({train_pct*100:.1f}%)")
        print(f"Validation: {len(self.val_data)} muestras ({val_pct*100:.1f}%)")
        print(f"Test: {len(self.test_data)} muestras ({(1-train_pct-val_pct)*100:.1f}%)")

    def create_windows(self) -> None:
        """
        Crea las ventanas de datos para cada conjunto usando los parámetros definidos
        en el constructor (input_width, label_width, shift)
        
        Nota:
            Debe haberse llamado previamente a split_data()
        """
        if self.train_data is None or self.val_data is None or self.test_data is None:
            raise ValueError("Debe dividir los datos primero usando split_data()")
        
        # Crear ventanas para cada conjunto
        self.train_windows = self.create_windows_for_set(self.train_data, self.total_window_size, self.input_slice, self.labels_slice)
        self.val_windows = self.create_windows_for_set(self.val_data, self.total_window_size, self.input_slice, self.labels_slice)
        self.test_windows = self.create_windows_for_set(self.test_data, self.total_window_size, self.input_slice, self.labels_slice)
        
        # Imprimir información sobre las ventanas creadas
        print("\nVentanas creadas:")
        print(f"Train: {len(self.train_windows[0])} ventanas")
        print(f"Validation: {len(self.val_windows[0])} ventanas")
        print(f"Test: {len(self.test_windows[0])} ventanas")
        print(f"\nTamaño de cada ventana:")
        print(f"Input: {self.input_width} pasos")
        print(f"Label: {self.label_width} pasos")

    def create_dataloaders(self, batch_size: int = 32, num_workers: int = 0) -> None:
        """
        Crea los DataLoaders de PyTorch para cada conjunto de ventanas
        
        Args:
            batch_size: Tamaño del batch (default: 32)
            num_workers: Número de workers para cargar datos (default: 0)
            
        Nota:
            Debe haberse llamado previamente a create_windows()
        """
        if self.train_windows is None or self.val_windows is None or self.test_windows is None:
            raise ValueError("Debe crear las ventanas primero usando create_windows()")
        
        # Crear datasets para cada conjunto
        train_dataset = WindowDataset(*self.train_windows)
        val_dataset = WindowDataset(*self.val_windows)
        test_dataset = WindowDataset(*self.test_windows)
        
        # Crear dataloaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,  # Mezclar aleatoriamente los datos de entrenamiento
            num_workers=num_workers
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,  # No mezclar datos de validación
            num_workers=num_workers
        )
        
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,  # No mezclar datos de test
            num_workers=num_workers
        )
        
        # Imprimir información sobre los dataloaders
        print("\nDataloaders creados:")
        print(f"Train: {len(train_dataset)} muestras, {len(self.train_loader)} batches")
        print(f"Validation: {len(val_dataset)} muestras, {len(self.val_loader)} batches")
        print(f"Test: {len(test_dataset)} muestras, {len(self.test_loader)} batches")
        print(f"\nConfiguración:")
        print(f"Batch size: {batch_size}")
        print(f"Workers: {num_workers}")
        print(f"Shuffle: True (train), False (val/test)")

    @staticmethod
    def create_windows_for_set(data: np.ndarray, 
                               total_window_size: int, 
                               input_slice: slice, 
                               labels_slice: slice,
                               label_columns: Optional[List[int]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Crea ventanas de input y labels para un conjunto de datos
        
        Args:
            data: Conjunto de datos a procesar
            total_window_size: Tamaño total de la ventana
            input_slice: Slice para la ventana de entrada
            labels_slice: Slice para la ventana de etiquetas
            label_columns: Columnas específicas para las etiquetas
            
        Returns:
            Tuple con (inputs, labels)
        """
        n_windows = len(data) - total_window_size + 1
        inputs = []
        labels = []
        
        for i in range(n_windows):
            window = data[i:i + total_window_size]
            input_window = window[input_slice]
            label_window = window[labels_slice]
            
            # Si hay columnas específicas para las etiquetas
            if label_columns is not None:
                label_window = label_window[:, label_columns]
            
            inputs.append(input_window)
            labels.append(label_window)
        
        return np.array(inputs), np.array(labels)
    
class DenseNN(nn.Module):
    """Red neuronal densa para predicción de series temporales"""
    def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int,
                 init_method: str = 'xavier_uniform', device: str = 'cpu'):
        """
        Inicializa la red neuronal densa
        
        Args:
            input_size: Tamaño de la entrada
            hidden_sizes: Lista con el número de neuronas para cada capa oculta
            output_size: Tamaño de la salida
            init_method: Método de inicialización de pesos ('xavier_uniform', 'xavier_normal', 
                        'kaiming_uniform', 'kaiming_normal', 'zeros', 'ones') (default: 'xavier_uniform')
            device: Dispositivo donde se ejecutará el modelo ('cpu' o 'cuda') (default: 'cpu')
        """
        super(DenseNN, self).__init__()
        self.device = device
        layers = []
        
        # Capa de entrada
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.ReLU())
        
        # Capas ocultas
        for i in range(len(hidden_sizes)-1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            layers.append(nn.ReLU())
        
        # Capa de salida
        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        
        # Crear red secuencial
        self.network = nn.Sequential(*layers)
        
        # Inicializar pesos
        self._init_weights(init_method)
        
        # Mover modelo al dispositivo especificado
        self.to(device)
        
    def _init_weights(self, method: str) -> None:
        """
        Inicializa los pesos de la red según el método especificado
        
        Args:
            method: Método de inicialización
        """
        for m in self.network.modules():
            if isinstance(m, nn.Linear):
                if method == 'xavier_uniform':
                    nn.init.xavier_uniform_(m.weight)
                elif method == 'xavier_normal':
                    nn.init.xavier_normal_(m.weight)
                elif method == 'kaiming_uniform':
                    nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                elif method == 'kaiming_normal':
                    nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                elif method == 'zeros':
                    nn.init.zeros_(m.weight)
                elif method == 'ones':
                    nn.init.ones_(m.weight)
                else:
                    raise ValueError(f"Método de inicialización desconocido: {method}")
                
                # Inicializar bias a cero
                nn.init.zeros_(m.bias)
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        return self.network(x)

class LSTM(nn.Module):
    """LSTM network for time series prediction"""
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int, device: str = 'cpu'):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.to(device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        # Ensure input has shape (batch_size, sequence_length, input_size)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # Add sequence dimension if missing
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

class CNN1D(nn.Module):
    """1D CNN for time series prediction"""
    def __init__(self, input_size: int, output_size: int, device: str = 'cpu'):
        super(CNN1D, self).__init__()
        self.device = device
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.fc1 = nn.Linear(64 * (input_size // 4), 128)
        self.fc2 = nn.Linear(128, output_size)
        self.to(device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        x = x.unsqueeze(1)
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class CNNLSTM(nn.Module):
    """Combined CNN and LSTM for time series prediction"""
    def __init__(self, input_size: int, hidden_size: int, output_size: int, device: str = 'cpu'):
        super(CNNLSTM, self).__init__()
        self.device = device
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.lstm = nn.LSTM(64, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.to(device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        x = x.unsqueeze(1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        _, (h_n, _) = self.lstm(x)
        x = self.fc(h_n[-1])
        return x

def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                optimizer: torch.optim.Optimizer, num_epochs: int, print_every: int = 10) -> Tuple[List[float], List[float]]:
    """
    Entrena un modelo PyTorch y evalúa en conjunto de validación
    
    Returns:
        Tuple conteniendo (train_losses, val_losses)
    """
    train_losses = []
    val_losses = []
    criterion = nn.MSELoss()
    device = next(model.parameters()).device  # Get the device the model is on
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        for inputs, targets in train_loader:
            # Move tensors to the same device as the model
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # Ensure outputs and targets have the same shape
            if outputs.shape != targets.shape:
                outputs = outputs.view(targets.shape)
                
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
        # Calcular loss de entrenamiento
        model.eval()
        epoch_loss = 0
        with torch.no_grad():
            for inputs, targets in train_loader:
                # Move tensors to the same device as the model
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                outputs = model(inputs)
                if outputs.shape != targets.shape:
                    outputs = outputs.view(targets.shape)
                loss = criterion(outputs, targets)
                epoch_loss += loss.item()
        train_losses.append(epoch_loss / len(train_loader))
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            val_epoch_loss = 0
            for inputs, targets in val_loader:
                # Move tensors to the same device as the model
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                outputs = model(inputs)
                if outputs.shape != targets.shape:
                    outputs = outputs.view(targets.shape)
                val_loss = criterion(outputs, targets)
                val_epoch_loss += val_loss.item()
            val_losses.append(val_epoch_loss / len(val_loader))
            
        if (epoch + 1) % print_every == 0:
            text = f'Epoch {epoch+1}/{num_epochs}:'
            text += f'Loss (Train/Val): {train_losses[-1]:.4f}/{val_losses[-1]:.4f}'
            print(text)
            
    return train_losses, val_losses