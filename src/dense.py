import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from typing import List, Optional, Tuple, Dict

class DenseNN(nn.Module):
    """Red neuronal densa para predicción de series temporales"""
    def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int,
                 init_method: str = 'xavier_uniform', device: str = 'cpu', dropout: float = 0.2):
        """
        Inicializa la red neuronal densa
        
        Args:
            input_size: Tamaño de la entrada
            hidden_sizes: Lista con el número de neuronas para cada capa oculta
            output_size: Tamaño de la salida
            init_method: Método de inicialización de pesos ('xavier_uniform', 'xavier_normal', 
                        'kaiming_uniform', 'kaiming_normal', 'zeros', 'ones') (default: 'xavier_uniform')
            device: Dispositivo donde se ejecutará el modelo ('cpu' o 'cuda') (default: 'cpu')
            dropout: Tasa de dropout entre capas (default: 0.2)
        """
        super(DenseNN, self).__init__()
        self.device = device
        layers = []
        
        # Capa de entrada
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(p=dropout))
        
        # Capas ocultas
        for i in range(len(hidden_sizes)-1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))
        
        # Capa de salida
        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        
        self.network = nn.Sequential(*layers)
        self._init_weights(init_method)
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

class DenseClassifier(DenseNN):
    """Red neuronal densa para problemas de clasificación"""
    def __init__(self, input_size: int, hidden_sizes: List[int], num_classes: int,
                 init_method: str = 'xavier_uniform', device: str = 'cpu', dropout: float = 0.2,
                 class_weights: Optional[torch.Tensor] = None):
        """
        Inicializa la red neuronal densa para clasificación
        
        Args:
            input_size: Tamaño de la entrada
            hidden_sizes: Lista con el número de neuronas para cada capa oculta
            num_classes: Número de clases para clasificación
            init_method: Método de inicialización de pesos
            device: Dispositivo donde se ejecutará el modelo
            dropout: Tasa de dropout entre capas
            class_weights: Pesos para cada clase para manejar desbalance
        """
        super(DenseClassifier, self).__init__(input_size, hidden_sizes, num_classes, 
                                            init_method, device, dropout)
        self.softmax = nn.LogSoftmax(dim=1)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        self.num_classes = num_classes
        
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return super().__call__(x)
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.softmax(self(x))
        _, predictions = torch.max(logits, dim=1)
        return predictions
    
    def compute_loss(self, data_loader: DataLoader) -> float:
        self.eval()
        total_loss = 0
        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                outputs = self(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
        return total_loss / len(data_loader)
    
    def compute_accuracy(self, data_loader: DataLoader) -> float:
        self.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                outputs = self(inputs)
                predictions = self.predict(outputs)
                correct += (predictions == targets).sum().item()
                total += targets.size(0)
        return correct / total
        
    def fit(self, train_loader: DataLoader, val_loader: DataLoader,
            optimizer: torch.optim.Optimizer, num_epochs: int, print_every: int = 10,
            metrics: Dict[str, List] = None) -> Dict[str, List]:
        """
        Entrena el modelo usando los datos proporcionados
        
        Args:
            train_loader: DataLoader con datos de entrenamiento
            val_loader: DataLoader con datos de validación
            optimizer: Optimizador a utilizar
            num_epochs: Número de épocas de entrenamiento
            print_every: Cada cuántas épocas imprimir resultados
            metrics: Diccionario con listas para almacenar epochs y métricas
        Returns:
            Dict con las métricas actualizadas
        """ 
        start_epoch = len(metrics['epochs'])
        
        for epoch in range(num_epochs):
            # Training
            self.train()
            for inputs, targets in train_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = self(inputs)
                train_loss = self.criterion(outputs, targets)
                train_loss.backward()
                optimizer.step()
                
            # Validation
            self.eval()
            train_epoch_loss = self.compute_loss(train_loader)
            val_epoch_loss = self.compute_loss(val_loader)
            train_epoch_accuracy = self.compute_accuracy(train_loader)
            val_epoch_accuracy = self.compute_accuracy(val_loader)
            
            metrics['epochs'].append(start_epoch + epoch + 1)
            metrics['loss']['train'].append(train_epoch_loss)
            metrics['loss']['val'].append(val_epoch_loss)
            metrics['accuracy']['train'].append(train_epoch_accuracy)
            metrics['accuracy']['val'].append(val_epoch_accuracy)
                
            if (epoch + 1) % print_every == 0:
                text = f'Epoch {epoch+1}/{num_epochs}: '
                text += f'Loss (Train/Val): {train_epoch_loss:.4f}/{val_epoch_loss:.4f}, '
                text += f'Acc (Train/Val): {train_epoch_accuracy:.2%}/{val_epoch_accuracy:.2%}'
                print(text)
                
        return metrics
    
    def compute_confusion_matrix(self, 
                                 data_loader: DataLoader, 
                                 plot: bool = True,
                                 save_path: Optional[str] = None,
                                 class_names: Optional[List[str]] = None
                                ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Calcula y opcionalmente visualiza la matriz de confusión para un conjunto de datos
        
        Args:
            data_loader: DataLoader con el conjunto de datos a evaluar
            plot: Si se debe visualizar la matriz de confusión
            save_path: Ruta opcional donde guardar la matriz de confusión
            class_names: Lista opcional con nombres de las clases
            
        Returns:
            Tupla con la matriz de confusión y diccionario de métricas
        """
        
        self.eval()
        y_true = []
        y_pred = []
        
        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs = inputs.to(self.device)
                outputs = self(inputs)
                _, predicted = torch.max(outputs.data, 1)
                y_true.extend(targets.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())
                
        # Calcular matriz y métricas
        cm = confusion_matrix(y_true, y_pred)
        metrics = {
            'f1_score': f1_score(y_true, y_pred, average='weighted'),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted')
        }
        
        if plot:
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=class_names if class_names else 'auto',
                       yticklabels=class_names if class_names else 'auto')
            plt.title('Matriz de Confusión')
            plt.xlabel('Predicción')
            plt.ylabel('Real')
            plt.tight_layout()
            if save_path:
                plt.savefig(save_path)
            plt.show()
            
        return cm, metrics