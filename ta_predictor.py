import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
import sys
import matplotlib
if(sys.argv[2] != 'p'):
  matplotlib.use('Agg') # Force non-interactive backend
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# --- 1. DATA PRE-PROCESSING ---


def ensure_full_year_15min_data(df, year):
    """Ensures every meter has every 15-minute interval for the full year[cite: 3]."""
    try:
      full_range = pd.date_range(start=f'{year}-01-01 00:00:00', 
                               end=f'{year}-12-31 23:45:00', 
                               freq='15T')
    except:
      full_range = pd.date_range(start=f'{year}-01-01 00:00:00', 
                               end=f'{year}-12-31 23:45:00', 
                               freq='15min')
    
    cleaned_list = []
    for meter_id in df['meter'].unique():
        meter_df = df[df['meter'] == meter_id].copy()
        meter_df = meter_df.set_index('x_Timestamp')
        meter_df = meter_df.reindex(full_range)
        meter_df['meter'] = meter_id
        meter_df['t_kWh'] = meter_df['t_kWh'].fillna(0.0)
        meter_df = meter_df.reset_index().rename(columns={'index': 'x_Timestamp'})
        cleaned_list.append(meter_df)
    return pd.concat(cleaned_list, ignore_index=True)

def process_data(ip_file, year):
    """Standard loader for the energy data[cite: 3]."""
    op_df = pd.read_csv(ip_file)
    op_df['x_Timestamp'] = pd.to_datetime(op_df['x_Timestamp'])
    op_df = ensure_full_year_15min_data(op_df, year)
    return op_df

# --- 2. TECHNICAL ANALYSIS FEATURE ENGINEERING ---

def add_ta_features(df):
    """
    Simulates Pandas TA functionality to create 'Stock Market' features.
    Includes RSI, Bollinger Bands, and EMA for energy momentum/volatility.
    """
    df = df.copy().sort_values('x_Timestamp')
    
    # 1. RSI (Relative Strength Index) - Momentum
    delta = df['t_kWh'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    df['RSI'] = 100 - (100 / (1 + rs))

    # 2. Bollinger Bands - Measures volatility and identifies spikes[cite: 1]
    df['MA20'] = df['t_kWh'].rolling(window=20).mean()
    df['STD20'] = df['t_kWh'].rolling(window=20).std()
    df['Upper_Band'] = df['MA20'] + (df['STD20'] * 2)
    df['Lower_Band'] = df['MA20'] - (df['STD20'] * 2)

    # 3. EMA (Exponential Moving Average) - High sensitivity to recent changes[cite: 1]
    df['EMA_short'] = df['t_kWh'].ewm(span=12, adjust=False).mean()
    
    # 4. Standard Lags for temporal continuity[cite: 3]
    df['lag_1'] = df['t_kWh'].shift(1)
    df['lag_2'] = df['t_kWh'].shift(2)
    
    # 5. Temporal Features[cite: 3]
    df['hour'] = df['x_Timestamp'].dt.hour
    df['day_of_week'] = df['x_Timestamp'].dt.dayofweek

    return df.dropna()

# --- 3. MAIN EXECUTION ---

# Load and prepare your datasets
if(sys.argv[1] == 'b'):
  train_full = process_data('Bareilly_2020_15t.csv', 2020)
  test_full = process_data('Bareilly_2021_15t.csv', 2021)
else:
  train_full = process_data('Mathura_2020_15t.csv', 2020)
  test_full = process_data('Mathura_2021_15t.csv', 2021)


# 2. Generate Clusters once
training_meters = train_full['meter'].unique()

r2_dict = {}
mae_dict = {}
for target_meter in training_meters:

  # Filter and generate TA features
  train_df = add_ta_features(train_full[train_full['meter'] == target_meter])
  test_df = add_ta_features(test_full[test_full['meter'] == target_meter])
  print(f"Processing Meter: {target_meter}")
  
  # Define features for the AdaBoost model
  features = ['RSI', 'Upper_Band', 'Lower_Band', 'EMA_short', 'lag_1', 'lag_2', 'hour', 'day_of_week']
  X_train, y_train = train_df[features], train_df['t_kWh']
  X_test, y_test = test_df[features], test_df['t_kWh']
  
  # Initialize and train AdaBoost[cite: 3]
  model = AdaBoostRegressor(
      DecisionTreeRegressor(max_depth=12), 
      n_estimators=125, 
      learning_rate=0.1,
      random_state=42
  )
  model.fit(X_train, y_train)
  
  # Generate predictions for 2021
  try:
    test_df['Prediction'] = model.predict(X_test)
  except ValueError:
    continue;
  # 1. Calculate Metrics
  mae = np.mean(np.abs(test_df['t_kWh'] - test_df['Prediction']))
  r2 = r2_score(test_df['t_kWh'], test_df['Prediction'])
  
  # 2. Print Results
  print("-" * 30)
  print(f"METER: {target_meter}")
  print(f"Mean Absolute Error: {mae:.4f} kWh")
  print(f"R^2 Score:           {r2:.4f}")
  print("-" * 30)
  
  # --- 4. VISUALIZATION ---
  
  plt.figure(figsize=(15, 6))
  # Select a 48-hour window (192 15-minute intervals) for clear viewing
  plot_df = test_df.iloc[:192*10] 
  
  plt.plot(plot_df['x_Timestamp'], plot_df['t_kWh'], label='Actual (2021)', color='#1f77b4', alpha=0.8)
  plt.plot(plot_df['x_Timestamp'], plot_df['Prediction'], label='TA-AdaBoost Prediction', color='#ff7f0e', linestyle='--')
  ax = plt.gca()
  
  # 1. Set the labels for every hour
  #ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
  #ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
  
  # 2. Create a tick for every 15-minute interval
  #ax.xaxis.set_minor_locator(mdates.MinuteLocator(byminute=[0, 15, 30, 45]))
  
  # 3. VERTICAL LINES ONLY: 
  # Set 'axis' to 'x' and 'which' to 'both' to catch the 15-min intervals
  plt.grid(True, which='both', axis='x', linestyle=':', color='gray', alpha=0.6)
  
  # Ensure horizontal lines are turned off
  plt.grid(False, axis='y')
  r2_dict[target_meter] = r2
  mae_dict[target_meter]= mae
  
  # 3. Optional: Add R^2 to the Plot Title
  plt.title(f"Energy Forecast {target_meter} | R^2: {r2:.3f}") 
  plt.xlabel("Time (HH:MM)")
  plt.ylabel("kWh")
  plt.legend()
  plt.xticks(rotation=45)
  plt.tight_layout()
  plt.savefig(f"Meter_{target_meter}_Forecast.png", dpi=300, bbox_inches='tight')
  if(sys.argv[2] == "p"):
    plt.show()# Quick Performance Metric
  plt.close()
  print(f"Prediction Complete. Mean Absolute Error: {mae:.4f} kWh")

print(r2_dict)
print(mae_dict)
