import calendar
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from typing import List, Tuple, Optional, Dict

# ======================================== #
#             Clases auxiliares            #
# ======================================== #

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
    
    def plot_window(self, 
                    idx: int = 0, 
                    figsize: Tuple[int, int] = (12, 6), 
                    save_path: Optional[str] = None) -> None:
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
        if save_path:
            plt.savefig(save_path)
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

    def split_data(self, train_pct: float = 0.7, val_pct: float = 0.15, input_width: int = 96) -> None:
        """
        Divide los datos en conjuntos de train, validation y test según porcentajes
        
        Args:
            train_pct: Porcentaje para entrenamiento (default: 0.7)
            val_pct: Porcentaje para validación (default: 0.15)
            input_width: Ancho de la ventana de entrada (default: 96)
            
        Nota:
            El porcentaje restante se usa para test
        """
        if train_pct + val_pct >= 1.0:
            raise ValueError("La suma de train_pct y val_pct debe ser menor a 1.0")
            
        n = len(self.data)
        train_end = int(n * train_pct)
        val_end = int(n * (train_pct + val_pct))
        
        # Incluir los últimos input_width datos del conjunto anterior
        self.train_data = self.data[:train_end]
        self.val_data = self.data[train_end-input_width:val_end] 
        self.test_data = self.data[val_end-input_width:]
        
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

class DataScaler:
    """
    Clase para escalar/normalizar datos de series temporales
    
    Attributes:
        scaler: Instancia de StandardScaler para el escalado
        feature_range: Rango deseado para el escalado (min, max)
    """
    def __init__(self, feature_range: Tuple[float, float] = (-1, 1)):
        """
        Inicializa el escalador
        
        Args:
            feature_range: Rango deseado para el escalado (min, max)
        """
        self.scaler = StandardScaler()
        self.feature_range = feature_range
        self.scale_ = None
        self.mean_ = None
        
    def fit(self, data: np.ndarray) -> None:
        """
        Ajusta el escalador a los datos
        
        Args:
            data: Array de datos a escalar
        """
        # Aplanar los datos para el ajuste
        original_shape = data.shape
        flattened_data = data.reshape(-1, 1)
        
        # Ajustar el escalador
        self.scaler.fit(flattened_data)
        
        # Guardar parámetros del escalador
        self.scale_ = self.scaler.scale_
        self.mean_ = self.scaler.mean_
        
    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Transforma los datos usando el escalador ajustado
        
        Args:
            data: Array de datos a transformar
            
        Returns:
            Array de datos escalados
        """
        # Aplanar los datos para la transformación
        original_shape = data.shape
        flattened_data = data.reshape(-1, 1)
        
        # Transformar los datos
        scaled_data = self.scaler.transform(flattened_data)
        
        # Reescalar al rango deseado
        min_val, max_val = self.feature_range
        scaled_data = min_val + (scaled_data - scaled_data.min()) * (max_val - min_val) / (scaled_data.max() - scaled_data.min())
        
        # Restaurar la forma original
        return scaled_data.reshape(original_shape)
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Invierte la transformación de los datos escalados
        
        Args:
            data: Array de datos escalados
            
        Returns:
            Array de datos en la escala original
        """
        # Aplanar los datos para la transformación inversa
        original_shape = data.shape
        flattened_data = data.reshape(-1, 1)
        
        # Invertir el reescalado al rango original
        min_val, max_val = self.feature_range
        data_original_range = (flattened_data - min_val) * (self.scale_ * 2) / (max_val - min_val) + self.mean_
        
        # Invertir la transformación del StandardScaler
        unscaled_data = self.scaler.inverse_transform(data_original_range)
        
        # Restaurar la forma original
        return unscaled_data.reshape(original_shape)
    
    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Ajusta el escalador y transforma los datos en un solo paso
        
        Args:
            data: Array de datos a escalar
            
        Returns:
            Array de datos escalados
        """
        self.fit(data)
        return self.transform(data)

class TrafficAnalyzer:
    """Class for analyzing traffic data and checking stationarity"""
    def __init__(self, data: pd.DataFrame):
        self.data = data
        
    def check_stationarity(self, series: pd.Series, n_diff: int = 0, pvalue_threshold: float = 0.05) -> str:
        """
        Perform Augmented Dickey-Fuller test with optional differencing and return results as string
        
        Args:
            series: Time series to test for stationarity
            n_diff: Number of times to difference the series before testing (default: 0)
            pvalue_threshold: Threshold for p-value to determine stationarity (default: 0.05)
            
        Returns:
            String containing test results and interpretation
        """
        # Apply differencing n times if requested
        diff_series = series.copy()
        for _ in range(n_diff):
            diff_series = diff_series.diff().dropna()
            
        result = adfuller(diff_series)
        is_stationary = result[1] < pvalue_threshold
        
        message = (
            f"Resultados del test de estacionariedad:\n"
            f"Estadístico ADF: {result[0]:.4f}\n"
            f"Valor p: {result[1]:.4f}\n"
            f"Valores críticos:\n"
        )
        
        for key, value in result[4].items():
            message += f"  {key}: {value:.4f}\n"
            
        message += f"La serie es {'estacionaria' if is_stationary else 'no estacionaria'}"
        
        return message
    
    def plot_autocorrelation(self, series: pd.Series, lags: int = 50, save_path: Optional[str] = None):
        """Plot autocorrelation and partial autocorrelation"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
        plot_acf(series, lags=lags, ax=ax1, auto_ylims=True)
        plot_pacf(series, lags=lags, ax=ax2, auto_ylims=True)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()

    def plot_time_series(
            self, 
            figsize: Tuple[int, int] = (15, 8), 
            columns: List[str] = ['Total'], 
            xticks_step: int = 96, 
            start_pct: float = 0, 
            end_pct: float = 100,
            title: str = 'Traffic Volume',
            xlabel: str = 'Date',
            ylabel: str = 'Traffic Volume',
            legend_loc: str = 'best',
            save_path: Optional[str] = None
            ):
        """Plot time series data with professional styling
        
        Args:
            columns: List of column names to plot. Defaults to ['Total']
            xticks_step: Number of 15-minute intervals between x-axis ticks. 
                        Default 96 represents 24 hours (96 * 15min = 24h)
            start_pct: Porcentaje inicial de la serie a mostrar (0-100). Default 0
            end_pct: Porcentaje final de la serie a mostrar (0-100). Default 100
            save_path: Ruta opcional donde guardar la gráfica
        """
        # Validar porcentajes
        if not 0 <= start_pct <= 100 or not 0 <= end_pct <= 100 or start_pct >= end_pct:
            raise ValueError("Los porcentajes deben estar entre 0 y 100, y start_pct debe ser menor que end_pct")
            
        # Calcular índices de inicio y fin
        n_total = len(self.data)
        start_idx = int(n_total * start_pct / 100)
        end_idx = int(n_total * end_pct / 100)
        
        # Seleccionar datos según el rango
        plot_data = self.data.iloc[start_idx:end_idx]
        
        plt.figure(figsize=figsize)
        
        # Plot each column with a different color
        colors = plt.cm.tab10(np.linspace(0, 1, len(columns)))
        for column, color in zip(columns, colors):
            plt.plot(plot_data['Datetime'], plot_data[column], linewidth=2, color=color, label=column)
        
        # Add title and labels
        if start_pct != 0 or end_pct != 100:
            title = f"{title} ({start_pct}% - {end_pct}%)"
        plt.title(title, fontsize=14, pad=15)
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        
        # Add legend
        plt.legend(fontsize=10, loc=legend_loc)
        
        # Customize grid and spines
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        
        # Set x-ticks every xticks_step intervals
        tick_positions = plot_data['Datetime'][::xticks_step]
        plt.xticks(tick_positions, [d.strftime('%Y-%m-%d %H:%M') for d in tick_positions], rotation=45, fontsize=10)
        plt.yticks(fontsize=10)
        
        # Add padding to prevent label cutoff
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()


# ======================================== #
#         Funciones en general             #
# ======================================== #

def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                optimizer: torch.optim.Optimizer, num_epochs: int, print_every: int = 10,
                loss: Dict[str, List] = None) -> Dict[str, List]:
    """
    Entrena un modelo PyTorch y evalúa en conjunto de validación
    
    Args:
        model: Modelo a entrenar
        train_loader: DataLoader con datos de entrenamiento
        val_loader: DataLoader con datos de validación  
        optimizer: Optimizador a utilizar
        num_epochs: Número de épocas de entrenamiento
        print_every: Cada cuántas épocas imprimir resultados
        loss: Diccionario con listas para almacenar epochs y losses
        
    Returns:
        Dict con epochs, train_losses y val_losses actualizados
    """
    if loss is None:
        loss = {'epochs': [], 'train': [], 'val': []}
    
    criterion = nn.MSELoss()
    device = next(model.parameters()).device  # Get the device the model is on
    
    start_epoch = len(loss['epochs'])
    
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
                
            train_loss = criterion(outputs, targets)
            train_loss.backward()
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
                batch_loss = criterion(outputs, targets)
                epoch_loss += batch_loss.item()
        
        loss['train'].append(epoch_loss / len(train_loader))
        
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
            
            loss['val'].append(val_epoch_loss / len(val_loader))
            loss['epochs'].append(start_epoch + epoch + 1)
            
        if (epoch + 1) % print_every == 0:
            text = f'Epoch {epoch+1}/{num_epochs}:'
            text += f'Loss (Train/Val): {loss["train"][-1]:.4f}/{loss["val"][-1]:.4f}'
            print(text)
            
    return loss

def evaluate_model(model: nn.Module, test_loader: DataLoader) -> float:
    """
    Evalúa el modelo en el conjunto de test
    
    Args:
        model: Modelo PyTorch a evaluar
        test_loader: DataLoader con los datos de test
        
    Returns:
        float: Valor de la loss en el conjunto de test
    """
    model.eval()
    criterion = nn.MSELoss()
    device = next(model.parameters()).device
    test_loss = 0.0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            # Move tensors to the same device as the model
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs)
            if outputs.shape != targets.shape:
                outputs = outputs.view(targets.shape)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            
    return test_loss / len(test_loader)

def get_predictions(model: nn.Module, test_loader: DataLoader) -> torch.Tensor:
    """
    Obtiene las predicciones del modelo para todo el conjunto de test
    
    Args:
        model: Modelo PyTorch entrenado
        test_loader: DataLoader con los datos de test
        
    Returns:
        torch.Tensor: Tensor con todas las predicciones en orden
    """
    model.eval()
    device = next(model.parameters()).device
    predictions = []
    
    with torch.no_grad():
        for inputs, _ in test_loader:
            # Move tensors to the same device as the model
            inputs = inputs.to(device)
            
            outputs = model(inputs)
            # Mover predicciones a CPU y convertir a numpy
            predictions.append(outputs.cpu())
            
    # Concatenar todas las predicciones
    return torch.cat(predictions, dim=0)

def plot_predictions(predictions: torch.Tensor, 
                    validation_data: np.ndarray,
                    test_data: np.ndarray,
                    samples_per_day: int = 96,
                    title: str = 'Predicciones vs Valores Reales',
                    save_path: Optional[str] = None) -> None:
    """
    Grafica las predicciones del modelo contra los valores reales
    
    Args:
        predictions: Tensor con las predicciones del modelo
        validation_data: Datos de validación
        test_data: Datos de test
        samples_per_day: Número de muestras por día
        title: Título de la gráfica
        save_path: Ruta opcional donde guardar la gráfica
    """
    # Convertir predicciones a numpy array y reshape
    predictions_np = predictions.numpy().reshape(-1)
    valid_values = validation_data.reshape(-1)[4*samples_per_day:]
    test_values = test_data.reshape(-1)[:len(predictions_np)]

    # Crear vectores de tiempo (samples_per_day valores por día)
    valid_tiempo = np.arange(0, len(valid_values)) / samples_per_day
    test_tiempo = np.arange(len(valid_values), len(valid_values) + len(test_values)) / samples_per_day
    pred_tiempo = test_tiempo.copy()

    # Graficar predicciones vs tiempo
    plt.figure(figsize=(10, 4))
    plt.axvspan(test_tiempo[0], test_tiempo[-1], color='lightgray', alpha=0.3)
    plt.plot(valid_tiempo, valid_values, 'g-', linewidth=1, label='Validación')
    plt.plot(test_tiempo, test_values, 'r-', linewidth=1, label='Test')
    plt.plot(pred_tiempo, predictions_np, 'b--', linewidth=1, label='Predicciones')

    # Configurar xticks cada samples_per_day datos (1 día)
    total_days = int(np.ceil(max(pred_tiempo)))
    plt.xticks(np.arange(0, total_days + 1, 1))

    plt.title(title)
    plt.xlabel('Días')
    plt.ylabel('Valores')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.show()


def plot_losses(history: Dict[str, List[float]], title: str, save_path: Optional[str] = None) -> None:
    """
    Grafica las pérdidas de entrenamiento y validación
    
    Args:
        history: Diccionario con las pérdidas {'epochs': [...], 'train': [...], 'val': [...]}
        title: Título de la gráfica
        save_path: Ruta opcional donde guardar la gráfica
    """
    plt.figure(figsize=(8, 4))
    plt.plot(history['epochs'], history['train'], label='Train')
    plt.plot(history['epochs'], history['val'], label='Validation')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, axis='both', linestyle='--', alpha=0.7)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight', facecolor='white')
    plt.show()

def plot_metrics(history: Dict[str, List[float]], title: str, save_path: Optional[str] = None) -> None:
    """
    Grafica las pérdidas y accuracy de entrenamiento y validación
    
    Args:
        history: Diccionario con métricas {'epochs': [...], 
                                         'loss': {'train': [...], 'val': [...]}, 
                                         'accuracy': {'train': [...], 'val': [...]}
                                         }
        title: Título de la gráfica
        save_path: Ruta opcional donde guardar la gráfica
    """
    fig, ax1 = plt.subplots(figsize=(8, 4))

    # Configurar eje izquierdo para loss
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_yscale('log')
    ln1 = ax1.plot(history['epochs'], history['loss']['train'], 'b-', label='Train Loss')
    ln2 = ax1.plot(history['epochs'], history['loss']['val'], 'r-', label='Val Loss')
    ax1.grid(True, axis='both', linestyle='--', alpha=0.7)

    # Configurar eje derecho para accuracy
    ax2 = ax1.twinx()
    ax2.set_ylabel('Accuracy')
    ln3 = ax2.plot(history['epochs'], history['accuracy']['train'], 'g--', label='Train Acc')
    ln4 = ax2.plot(history['epochs'], history['accuracy']['val'], 'y--', label='Val Acc')

    # Combinar leyendas de ambos ejes
    lns = ln1 + ln2 + ln3 + ln4
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='center right')

    plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight', facecolor='white')
    plt.show()

def create_datetime_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a complete datetime column combining Time, Date and Day of the week.
    Assumes data starts in March 2025 and handles month transitions.
    
    Args:
        df (pd.DataFrame): DataFrame containing Time, Date, and Day of the week columns
        
    Returns:
        pd.DataFrame: DataFrame with added complete_datetime column
    """
    # Convert Time to datetime format for proper handling
    df['Time'] = pd.to_datetime(df['Time'], format='%I:%M:%S %p').dt.strftime('%H:%M:%S')
    
    # Initialize first date as March 2025
    start_year = 2025
    start_month = 3
    
    # Create list to store complete dates
    complete_dates = []
    current_month = start_month
    current_year = start_year
    prev_date = None
    
    # Get number of days in each month
    month_days = {month: calendar.monthrange(current_year, month)[1] for month in range(1, 13)}
    
    for _, row in df.iterrows():
        day = int(row['Date'])
        
        # Check if we need to move to next month
        if prev_date is not None and day < prev_date:
            current_month += 1
            if current_month > 12:
                current_month = 1
                current_year += 1
                # Update month_days for the new year
                month_days = {month: calendar.monthrange(current_year, month)[1] for month in range(1, 13)}
        
        # Ensure day is valid for the current month
        if day > month_days[current_month]:
            day = month_days[current_month]
        
        # Create complete datetime string
        date_str = f"{current_year}-{current_month:02d}-{day:02d} {row['Time']}"
        complete_dates.append(pd.to_datetime(date_str))
        
        prev_date = day
    
    # Add new column
    df['Datetime'] = complete_dates
    
    return df