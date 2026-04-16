import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def create_extraction_datasets():
    print("Loading all window datasets...")
    train_df = pd.read_csv('train_windows.csv')
    val_df = pd.read_csv('val_windows.csv')
    test_df = pd.read_csv('test_windows.csv')
    
    # Combine all datasets
    all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    print(f"Total windows: {len(all_df)}")
    
    # Filter to only pump events (pump=1)
    pump_df = all_df[all_df['Pump'] == 1].copy()
    print(f"Total pump events: {len(pump_df)}")
    
    # Convert date column to datetime and sort by date (earliest first)
    pump_df['date'] = pd.to_datetime(pump_df['date'], errors='coerce')
    pump_df = pump_df.sort_values('date').reset_index(drop=True)
    
    # Remove rows with invalid dates
    pump_df = pump_df.dropna(subset=['date'])
    print(f"Pump events with valid dates: {len(pump_df)}")
    
    # Create 60/20/20 split based on chronological order
    total_pump = len(pump_df)
    train_size = int(0.6 * total_pump)
    val_size = int(0.2 * total_pump)
    
    train_df = pump_df.iloc[:train_size].copy()
    val_df = pump_df.iloc[train_size:train_size + val_size].copy()
    test_df = pump_df.iloc[train_size + val_size:].copy()
    
    print(f"\n60/20/20 Split Results (chronological order):")
    print(f"Train: {len(train_df)} pump events (60%) - Date range: {train_df['date'].min()} to {train_df['date'].max()}")
    print(f"Val: {len(val_df)} pump events (20%) - Date range: {val_df['date'].min()} to {val_df['date'].max()}")
    print(f"Test: {len(test_df)} pump events (20%) - Date range: {test_df['date'].min()} to {test_df['date'].max()}")
    
    # Save extraction datasets
    train_df.to_csv('train_extraction.csv', index=False)
    val_df.to_csv('val_extraction.csv', index=False)
    test_df.to_csv('test_extraction.csv', index=False)
    
    print(f"\nSaved extraction datasets:")
    print(f"- train_extraction.csv: {len(train_df)} pump events")
    print(f"- val_extraction.csv: {len(val_df)} pump events")
    print(f"- test_extraction.csv: {len(test_df)} pump events")
    
    # Print some statistics about the extraction datasets
    print(f"\nExtraction Dataset Statistics:")
    total_pump = len(train_df) + len(val_df) + len(test_df)
    print(f"Total pump events: {total_pump}")
    print(f"Train/Val/Test split: {len(train_df)}/{len(val_df)}/{len(test_df)}")
    
    # Check for missing coin/exchange data
    train_missing_coin = train_df['coin_cleaned'].isna().sum()
    train_missing_exchange = train_df['exchange_cleaned'].isna().sum()
    val_missing_coin = val_df['coin_cleaned'].isna().sum()
    val_missing_exchange = val_df['exchange_cleaned'].isna().sum()
    test_missing_coin = test_df['coin_cleaned'].isna().sum()
    test_missing_exchange = test_df['exchange_cleaned'].isna().sum()
    
    print(f"\nMissing data:")
    print(f"Train - Missing coin: {train_missing_coin}, Missing exchange: {train_missing_exchange}")
    print(f"Val - Missing coin: {val_missing_coin}, Missing exchange: {val_missing_exchange}")
    print(f"Test - Missing coin: {test_missing_coin}, Missing exchange: {test_missing_exchange}")

if __name__ == "__main__":
    create_extraction_datasets() 