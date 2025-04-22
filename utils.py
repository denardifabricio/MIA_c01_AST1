import pandas as pd
import calendar

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