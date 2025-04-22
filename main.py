import pandas as pd
import numpy as np
from model import (
    TrafficAnalyzer, TimeSeriesDataset, DenseNN, LSTM, CNN1D, CNNLSTM,
    train_model, evaluate_arima, evaluate_sarima
)
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from datetime import datetime
import calendar

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the traffic data"""
    # Initialize variables
    current_year = datetime.now().year
    current_month = 1  # Start with January
    complete_dates = []
    
    # Get the number of days in each month
    month_days = {month: calendar.monthrange(current_year, month)[1] for month in range(1, 13)}
    
    # Convert time to 24-hour format
    df['Time'] = pd.to_datetime(df['Time'], format='%I:%M:%S %p').dt.strftime('%H:%M:%S')
    
    # Process each row
    prev_day = None
    for idx, row in df.iterrows():
        day = int(row['Date'])
        
        # Handle month transitions
        if prev_day is not None and day < prev_day:
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
        
        prev_day = day
    
    # Add the new datetime column
    df['timestamp'] = complete_dates
    df.set_index('timestamp', inplace=True)
    
    # One-hot encode day of week
    df = pd.get_dummies(df, columns=['Day of the week'])
    
    return df

def main():
    # Load and preprocess data
    df = pd.read_csv('traffic_data.csv')  # Replace with your actual data file
    df = preprocess_data(df)
    
    # Initialize analyzer
    analyzer = TrafficAnalyzer(df)
    
    # Analyze stationarity
    for column in ['Total', 'CarCount', 'BikeCount', 'BusCount', 'TruckCount']:
        test_stat, p_value, critical_values = analyzer.check_stationarity(df[column])
        print(f"\n{column} Stationarity Test:")
        print(f"ADF Statistic: {test_stat}")
        print(f"p-value: {p_value}")
        print("Critical Values:", critical_values)
        
        # Plot autocorrelation
        analyzer.plot_autocorrelation(df[column])
    
    # Prepare data for deep learning models
    target_column = 'Total'  # or any other vehicle count column
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[[target_column]])
    
    # Create dataset
    window_size = 24  # 6 hours of 15-minute intervals
    horizon = 4  # 1 hour ahead prediction
    dataset = TimeSeriesDataset(scaled_data, window_size, horizon)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Initialize models
    input_size = window_size
    hidden_size = 64
    output_size = horizon
    
    models = {
        'DenseNN': DenseNN(input_size, hidden_size, output_size),
        'LSTM': LSTM(input_size, hidden_size, 2, output_size),
        'CNN1D': CNN1D(input_size, output_size),
        'CNNLSTM': CNNLSTM(input_size, hidden_size, output_size)
    }
    
    # Train and evaluate deep learning models
    criterion = nn.MSELoss()
    results = {}
    
    for name, model in models.items():
        optimizer = torch.optim.Adam(model.parameters())
        losses = train_model(model, train_loader, criterion, optimizer, num_epochs=50)
        results[name] = losses
        
        # Plot training loss
        plt.figure(figsize=(10, 6))
        plt.plot(losses)
        plt.title(f'{name} Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.show()
    
    # Evaluate ARIMA models
    arima_results = []
    for p in range(3):
        for d in range(2):
            for q in range(3):
                aic, bic = evaluate_arima(df[target_column], p, d, q)
                arima_results.append({
                    'p': p,
                    'd': d,
                    'q': q,
                    'AIC': aic,
                    'BIC': bic
                })
    
    # Evaluate SARIMA models
    sarima_results = []
    seasonal_period = 96  # 24 hours in 15-minute intervals
    for p in range(2):
        for d in range(2):
            for q in range(2):
                for P in range(2):
                    for D in range(2):
                        for Q in range(2):
                            aic, bic = evaluate_sarima(
                                df[target_column],
                                (p, d, q),
                                (P, D, Q, seasonal_period)
                            )
                            sarima_results.append({
                                'p': p,
                                'd': d,
                                'q': q,
                                'P': P,
                                'D': D,
                                'Q': Q,
                                'AIC': aic,
                                'BIC': bic
                            })
    
    # Print best models
    arima_df = pd.DataFrame(arima_results)
    sarima_df = pd.DataFrame(sarima_results)
    
    print("\nBest ARIMA Model:")
    print(arima_df.loc[arima_df['AIC'].idxmin()])
    
    print("\nBest SARIMA Model:")
    print(sarima_df.loc[sarima_df['AIC'].idxmin()])

if __name__ == "__main__":
    main() 