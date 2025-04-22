import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook
from itertools import product
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import List, Tuple, Optional, Union, Dict


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
    
    def plot_autocorrelation(self, series: pd.Series, lags: int = 50):
        """Plot autocorrelation and partial autocorrelation"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
        plot_acf(series, lags=lags, ax=ax1, auto_ylims=True)
        plot_pacf(series, lags=lags, ax=ax2, auto_ylims=True)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
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
            legend_loc: str = 'best'
            ):
        """Plot time series data with professional styling
        
        Args:
            columns: List of column names to plot. Defaults to ['Total']
            xticks_step: Number of 15-minute intervals between x-axis ticks. 
                        Default 96 represents 24 hours (96 * 15min = 24h)
            start_pct: Porcentaje inicial de la serie a mostrar (0-100). Default 0
            end_pct: Porcentaje final de la serie a mostrar (0-100). Default 100
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
        
        plt.show()

class ARIMAModel:
    """Clase para manejar modelos ARIMA con funcionalidades de evaluación, optimización y predicción"""
    
    def __init__(self, data: pd.Series):
        """
        Inicializa el modelo ARIMA
        
        Args:
            data: Serie temporal para el modelo
        """
        self.data = data
        self.model = None
        self.results = None
        self.best_order = None
        
    def plot_data(self, figsize: Tuple[int, int] = (15, 6), 
                 title: str = 'Serie Temporal',
                 xlabel: str = 'Días',
                 ylabel: str = 'Valor',
                 tick_interval: int = 96) -> None:
        """
        Grafica la serie temporal con ticks personalizados en el eje x
        
        Args:
            figsize: Tamaño de la figura (ancho, alto)
            title: Título del gráfico
            xlabel: Etiqueta del eje x
            ylabel: Etiqueta del eje y
            tick_interval: Intervalo para los ticks del eje x (debe ser múltiplo de 96)
        """
        if tick_interval % 96 != 0:
            raise ValueError("tick_interval debe ser múltiplo de 96")
            
        plt.figure(figsize=figsize)
        
        # Graficar la serie
        plt.plot(self.data.values, linewidth=1.5)
        
        # Personalizar el eje x
        n_days = len(self.data) // 96  # Número de días completos
        day_step = tick_interval // 96  # Cada cuántos días mostrar tick
        xticks_positions = [i * tick_interval for i in range(n_days // day_step + 1)]
        xticks_labels = [str(i * day_step) for i in range(n_days // day_step + 1)]
        
        plt.xticks(xticks_positions, xticks_labels)
        
        # Personalizar el gráfico
        plt.title(title, fontsize=14, pad=15)
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Ocultar spines superior y derecho
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        
        plt.tight_layout()
        plt.show()

    def evaluate(self, p: int, d: int, q: int) -> Tuple[float, float, bool, str]:
        """
        Evalúa un modelo ARIMA con parámetros específicos
        
        Args:
            p: Orden AR
            d: Orden de diferenciación
            q: Orden MA
            
        Returns:
            Tuple con (AIC, BIC, estado de convergencia, mensaje de error si hay)
        """
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                
                model = ARIMA(
                    self.data,
                    order=(p, d, q),
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
                
                results = model.fit()
                return results.aic, results.bic, True, ""
                
        except Exception as e:
            return float('inf'), float('inf'), False, str(e)
            
    def optimize(self, p_range: range, d: int, q_range: range) -> pd.DataFrame:
        """
        Optimiza el modelo ARIMA probando diferentes combinaciones de parámetros
        
        Args:
            p_range: Rango de valores p a probar
            d: Orden de diferenciación
            q_range: Rango de valores q a probar
            
        Returns:
            DataFrame con resultados ordenados por AIC
        """
        results = []
        orders = list(product(p_range, [d], q_range))
        
        for order in tqdm_notebook(orders, desc="Optimizando ARIMA"):
            p, d, q = order
            aic, bic, converged, error = self.evaluate(p, d, q)
            
            if not converged:
                print(f"Error al ajustar ARIMA{order}: {error}")
                
            results.append({
                'order': f"({p},{d},{q})",
                'aic': aic,
                'bic': bic,
                'converged': converged,
                'error': error if not converged else ""
            })
            
        df_results = pd.DataFrame(results)
        
        # Imprimir resumen
        n_successful = df_results['converged'].sum()
        n_total = len(df_results)
        print(f"\nResumen de optimización:")
        print(f"Ajustes exitosos: {n_successful}/{n_total} ({n_successful/n_total*100:.1f}%)")
        
        if n_successful > 0:
            best_model = df_results[df_results['converged']].sort_values('aic').iloc[0]
            self.best_order = eval(best_model['order'])  # Guardar mejor orden
            print(f"\nMejor modelo: ARIMA{best_model['order']}")
            print(f"AIC: {best_model['aic']:.2f}")
            print(f"BIC: {best_model['bic']:.2f}")
        
        return df_results.sort_values('aic')
        
    def fit(self, order: Optional[Tuple[int, int, int]] = None) -> None:
        """
        Ajusta el modelo ARIMA con los parámetros especificados o los mejores encontrados
        
        Args:
            order: Orden del modelo (p,d,q). Si None, usa el mejor orden encontrado
        """
        if order is None and self.best_order is None:
            raise ValueError("Debe especificar el orden o ejecutar optimize() primero")
            
        order_to_use = order if order is not None else self.best_order
        
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            self.model = ARIMA(self.data, order=order_to_use)
            self.results = self.model.fit()
            
    def rolling_forecast(self, train_size: int, window: int = 1) -> List[float]:
        """
        Realiza predicciones rolling forecast
        
        Args:
            train_size: Tamaño del conjunto de entrenamiento
            window: Tamaño de la ventana de predicción
            
        Returns:
            Lista con las predicciones
        """
        if self.model is None:
            raise ValueError("Debe ajustar el modelo primero usando fit()")
            
        predictions = []
        history = list(self.data[:train_size])
        total_len = train_size + len(self.data[train_size:])
        
        iters = range(train_size, total_len, window)
        for i in tqdm_notebook(iters, desc="Rolling forecast ARIMA"):
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                
                model = ARIMA(history[:i], order=self.model.order)
                res = model.fit()
                pred = res.get_prediction(i, i + window - 1)
                yhat = pred.predicted_mean
                predictions += list(yhat)
                
                if i < len(self.data):
                    history.append(self.data.iloc[i])
        
        horizon = len(self.data[train_size:])
        if len(predictions) > horizon:
            predictions = predictions[:horizon]
            
        return predictions

    def plot_predictions(self, train_data: Union[np.ndarray, List[float]], 
                       test_data: Union[np.ndarray, List[float]], 
                       predictions: Union[np.ndarray, List[float]],
                       figsize: Tuple[int, int] = (15, 6),
                       title: str = 'Predicciones vs Real',
                       xlabel: str = 'Días',
                       ylabel: str = 'Valor',
                       tick_interval: int = 96) -> None:
        """
        Grafica los datos de entrenamiento, test y predicciones
        
        Args:
            train_data: Datos de entrenamiento
            test_data: Datos reales de test
            predictions: Predicciones del modelo
            figsize: Tamaño de la figura
            title: Título del gráfico
            xlabel: Etiqueta del eje x
            ylabel: Etiqueta del eje y
            tick_interval: Intervalo para los ticks del eje x (debe ser múltiplo de 96)
        """
        if tick_interval % 96 != 0:
            raise ValueError("tick_interval debe ser múltiplo de 96")
            
        if len(test_data) != len(predictions):
            raise ValueError("Los datos de test y las predicciones deben tener la misma longitud")
            
        plt.figure(figsize=figsize)
        
        # Convertir a arrays numpy si son listas
        train_data = np.array(train_data)
        test_data = np.array(test_data)
        predictions = np.array(predictions)
        
        # Crear el eje x
        x_train = np.arange(len(train_data))
        x_test = np.arange(len(train_data), len(train_data) + len(test_data))
        
        plt.plot(x_train, train_data, 'b-', label='Entrenamiento', linewidth=1.5)
        plt.plot(x_test, test_data, 'g-', label='Test (Real)', linewidth=1.5)
        plt.plot(x_test, predictions, 'r--', label='Predicciones', linewidth=1.5)
        
        # Añadir fondo gris para la zona de test
        plt.axvspan(x_test[0], x_test[-1], color='gray', alpha=0.1)
        
        # Personalizar el eje x
        total_days = (len(train_data) + len(test_data)) // 96
        day_step = tick_interval // 96
        xticks_positions = [i * tick_interval for i in range(total_days // day_step + 1)]
        xticks_labels = [str(i * day_step) for i in range(total_days // day_step + 1)]
        
        plt.xticks(xticks_positions, xticks_labels)
        
        # Personalizar el gráfico
        plt.title(title, fontsize=14, pad=15)
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=10, loc='best')
        
        # Ocultar spines superior y derecho
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        
        plt.tight_layout()
        plt.show()

    def calculate_metrics(self, test_data: Union[np.ndarray, List[float]], 
                        predictions: Union[np.ndarray, List[float]]) -> Dict[str, float]:
        """
        Calcula métricas de error para las predicciones
        
        Args:
            test_data: Datos reales de test
            predictions: Predicciones del modelo
            
        Returns:
            Diccionario con las métricas calculadas
        """
        if len(test_data) != len(predictions):
            raise ValueError("Los datos de test y las predicciones deben tener la misma longitud")
            
        # Convertir a arrays numpy si son listas
        test_data = np.array(test_data)
        predictions = np.array(predictions)
        
        # Calcular métricas
        rmse = np.sqrt(mean_squared_error(test_data, predictions))
        mae = mean_absolute_error(test_data, predictions)
        mape = np.mean(np.abs((test_data - predictions) / test_data)) * 100
        
        return {
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape
        }
        
    def print_metrics(self, test_data: Union[np.ndarray, List[float]], 
                     predictions: Union[np.ndarray, List[float]]) -> None:
        """
        Imprime las métricas de error de forma formateada
        
        Args:
            test_data: Datos reales de test
            predictions: Predicciones del modelo
        """
        metrics = self.calculate_metrics(test_data, predictions)
        
        print("\nMétricas de Error:")
        print("-" * 30)
        print(f"RMSE: {metrics['RMSE']:.4f}")
        print(f"MAE:  {metrics['MAE']:.4f}")
        print(f"MAPE: {metrics['MAPE']:.2f}%")

class SARIMAModel:
    """Clase para manejar modelos SARIMA con funcionalidades de evaluación, optimización y predicción"""
    
    def __init__(self, data: pd.Series):
        """
        Inicializa el modelo SARIMA
        
        Args:
            data: Serie temporal para el modelo
        """
        self.data = data
        self.model = None
        self.results = None
        self.best_order = None
        self.best_seasonal_order = None
        
    def plot_data(self, figsize: Tuple[int, int] = (15, 6), 
                 title: str = 'Serie Temporal',
                 xlabel: str = 'Días',
                 ylabel: str = 'Valor',
                 tick_interval: int = 96) -> None:
        """
        Grafica la serie temporal con ticks personalizados en el eje x
        
        Args:
            figsize: Tamaño de la figura (ancho, alto)
            title: Título del gráfico
            xlabel: Etiqueta del eje x
            ylabel: Etiqueta del eje y
            tick_interval: Intervalo para los ticks del eje x (debe ser múltiplo de 96)
        """
        if tick_interval % 96 != 0:
            raise ValueError("tick_interval debe ser múltiplo de 96")
            
        plt.figure(figsize=figsize)
        
        # Graficar la serie
        plt.plot(self.data.values, linewidth=1.5)
        
        # Personalizar el eje x
        n_days = len(self.data) // 96  # Número de días completos
        day_step = tick_interval // 96  # Cada cuántos días mostrar tick
        xticks_positions = [i * tick_interval for i in range(n_days // day_step + 1)]
        xticks_labels = [str(i * day_step) for i in range(n_days // day_step + 1)]
        
        plt.xticks(xticks_positions, xticks_labels)
        
        # Personalizar el gráfico
        plt.title(title, fontsize=14, pad=15)
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Ocultar spines superior y derecho
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        
        plt.tight_layout()
        plt.show()

    def evaluate(self, order: Tuple[int, int, int], seasonal_order: Tuple[int, int, int, int]) -> Tuple[float, float, bool, str]:
        """
        Evalúa un modelo SARIMA específico
        
        Args:
            order: Tuple de (p,d,q) para parámetros no estacionales
            seasonal_order: Tuple de (P,D,Q,m) para parámetros estacionales
            
        Returns:
            Tuple con (AIC, BIC, estado de convergencia, mensaje de error si hay)
        """
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                
                model = SARIMAX(
                    self.data,
                    order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
                
                results = model.fit()
                return results.aic, results.bic, True, ""
                
        except Exception as e:
            return float('inf'), float('inf'), False, str(e)
            
    def optimize(self, p_range: range, d: int, q_range: range,
                P_range: range, D: int, Q_range: range, m: int) -> pd.DataFrame:
        """
        Optimiza el modelo SARIMA probando diferentes combinaciones de parámetros
        
        Args:
            p_range: Range of p values to try
            d: Differencing order
            q_range: Range of q values to try
            P_range: Range of seasonal P values to try
            D: Seasonal differencing order
            Q_range: Range of seasonal Q values to try
            m: Seasonal period
            
        Returns:
            DataFrame with columns [(p,d,q)(P,D,Q)m, AIC, BIC] sorted by AIC
        """
        results = []
        orders = list(product(p_range, [d], q_range))
        seasonal_orders = list(product(P_range, [D], Q_range, [m]))
        for order in tqdm_notebook(orders, desc="Optimizing SARIMA"):
            for seasonal_order in tqdm_notebook(seasonal_orders, desc="Seasonal loop"):
                aic, bic, converged, error = self.evaluate(order, seasonal_order)
                
                if not converged:
                    print(f"Failed to fit SARIMA{order}{seasonal_order}: {error}")
                    
                p, d, q = order
                P, D, Q, m = seasonal_order
                results.append({
                    'order': f"({p},{d},{q})",
                    'seasonal_order': f"({P},{D},{Q},{m})",
                    'aic': aic,
                    'bic': bic,
                    'converged': converged,
                    'error': error if not converged else ""
                })
                    
        # Convert to DataFrame and sort by AIC
        df_results = pd.DataFrame(results)
        
        # Print summary of results
        n_successful = df_results['converged'].sum()
        n_total = len(df_results)
        print(f"\nOptimization Summary:")
        print(f"Successful fits: {n_successful}/{n_total} ({n_successful/n_total*100:.1f}%)")
        
        if n_successful > 0:
            best_model = df_results[df_results['converged']].sort_values('aic').iloc[0]
            self.best_order = eval(best_model['order'])  # Guardar mejor orden (sin m)
            self.best_seasonal_order = eval(best_model['seasonal_order'])  # Guardar mejor orden estacional
            print(f"\nMejor modelo: SARIMA{best_model['order']}{best_model['seasonal_order']}")
            print(f"AIC: {best_model['aic']:.2f}")
            print(f"BIC: {best_model['bic']:.2f}")
        
        return df_results.sort_values('aic')
        
    def fit(self, order: Tuple[int, int, int], seasonal_order: Tuple[int, int, int, int]) -> None:
        """
        Ajusta el modelo SARIMA con los parámetros especificados
        
        Args:
            order: Tuple de (p,d,q) para parámetros no estacionales
            seasonal_order: Tuple de (P,D,Q,m) para parámetros estacionales
        """
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            self.model = SARIMAX(self.data, order=order, seasonal_order=seasonal_order)
            self.results = self.model.fit()
            
    def rolling_forecast(self, train_size: int, window: int = 1) -> List[float]:
        """
        Realiza predicciones rolling forecast
        
        Args:
            train_size: Tamaño del conjunto de entrenamiento
            window: Tamaño de la ventana de predicción
            
        Returns:
            Lista con las predicciones
        """
        if self.model is None:
            raise ValueError("Debe ajustar el modelo primero usando fit()")
            
        predictions = []
        history = list(self.data[:train_size])
        total_len = train_size + len(self.data[train_size:])
        
        iters = range(train_size, total_len, window)
        for i in tqdm_notebook(iters, desc="Rolling forecast SARIMA"):
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                
                model = SARIMAX(history[:i], order=self.best_order, seasonal_order=self.best_seasonal_order)
                res = model.fit()
                pred = res.get_prediction(i, i + window - 1)
                yhat = pred.predicted_mean
                predictions += list(yhat)
                
                if i < len(self.data):
                    history.append(self.data.iloc[i])
        
        horizon = len(self.data[train_size:])
        if len(predictions) > horizon:
            predictions = predictions[:horizon]
            
        return predictions

    def plot_predictions(self, train_data: Union[np.ndarray, List[float]], 
                       test_data: Union[np.ndarray, List[float]], 
                       predictions: Union[np.ndarray, List[float]],
                       figsize: Tuple[int, int] = (15, 6),
                       title: str = 'Predicciones vs Real',
                       xlabel: str = 'Días',
                       ylabel: str = 'Valor',
                       tick_interval: int = 96) -> None:
        """
        Grafica los datos de entrenamiento, test y predicciones
        
        Args:
            train_data: Datos de entrenamiento
            test_data: Datos reales de test
            predictions: Predicciones del modelo
            figsize: Tamaño de la figura
            title: Título del gráfico
            xlabel: Etiqueta del eje x
            ylabel: Etiqueta del eje y
            tick_interval: Intervalo para los ticks del eje x (debe ser múltiplo de 96)
        """
        if tick_interval % 96 != 0:
            raise ValueError("tick_interval debe ser múltiplo de 96")
            
        if len(test_data) != len(predictions):
            raise ValueError("Los datos de test y las predicciones deben tener la misma longitud")
            
        plt.figure(figsize=figsize)
        
        # Convertir a arrays numpy si son listas
        train_data = np.array(train_data)
        test_data = np.array(test_data)
        predictions = np.array(predictions)
        
        # Crear el eje x
        x_train = np.arange(len(train_data))
        x_test = np.arange(len(train_data), len(train_data) + len(test_data))
        
        plt.plot(x_train, train_data, 'b-', label='Entrenamiento', linewidth=1.5)
        plt.plot(x_test, test_data, 'g-', label='Test (Real)', linewidth=1.5)
        plt.plot(x_test, predictions, 'r--', label='Predicciones', linewidth=1.5)
        
        # Añadir fondo gris para la zona de test
        plt.axvspan(x_test[0], x_test[-1], color='gray', alpha=0.1)
        
        # Personalizar el eje x
        total_days = (len(train_data) + len(test_data)) // 96
        day_step = tick_interval // 96
        xticks_positions = [i * tick_interval for i in range(total_days // day_step + 1)]
        xticks_labels = [str(i * day_step) for i in range(total_days // day_step + 1)]
        
        plt.xticks(xticks_positions, xticks_labels)
        
        # Personalizar el gráfico
        plt.title(title, fontsize=14, pad=15)
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=10, loc='best')
        
        # Ocultar spines superior y derecho
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        
        plt.tight_layout()
        plt.show()

    def calculate_metrics(self, test_data: Union[np.ndarray, List[float]], 
                        predictions: Union[np.ndarray, List[float]]) -> Dict[str, float]:
        """
        Calcula métricas de error para las predicciones
        
        Args:
            test_data: Datos reales de test
            predictions: Predicciones del modelo
            
        Returns:
            Diccionario con las métricas calculadas
        """
        if len(test_data) != len(predictions):
            raise ValueError("Los datos de test y las predicciones deben tener la misma longitud")
            
        # Convertir a arrays numpy si son listas
        test_data = np.array(test_data)
        predictions = np.array(predictions)
        
        # Calcular métricas
        rmse = np.sqrt(mean_squared_error(test_data, predictions))
        mae = mean_absolute_error(test_data, predictions)
        mape = np.mean(np.abs((test_data - predictions) / test_data)) * 100
        
        return {
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape
        }
        
    def print_metrics(self, test_data: Union[np.ndarray, List[float]], 
                     predictions: Union[np.ndarray, List[float]]) -> None:
        """
        Imprime las métricas de error de forma formateada
        
        Args:
            test_data: Datos reales de test
            predictions: Predicciones del modelo
        """
        metrics = self.calculate_metrics(test_data, predictions)
        
        print("\nMétricas de Error:")
        print("-" * 30)
        print(f"RMSE: {metrics['RMSE']:.4f}")
        print(f"MAE:  {metrics['MAE']:.4f}")
        print(f"MAPE: {metrics['MAPE']:.2f}%")
