import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pywt
from scipy.signal import find_peaks

def convert_to_15min_data(ip_file, op_file):
  df = pd.read_csv(ip_file)
  df['x_Timestamp'] = pd.to_datetime(df['x_Timestamp'])
  df.set_index('x_Timestamp', inplace=True)

  # 15-Minute Aggregation Logic
  agg_logic = {
      't_kWh': 'sum',
      'z_Avg Voltage (Volt)': 'mean',
      'z_Avg Current (Amp)': 'mean',
      'y_Freq (Hz)': 'mean',
      'meter': 'first' # Keep the meter ID
  }
  # Resample to 15-minute blocks
  df_15min = df.groupby('meter').resample('15T').agg(agg_logic)
  
  # Reset index to get a clean dataframe
  df_15min = df_15min.drop(columns=['meter']).reset_index()
  print(df_15min.head())
  df_15min.to_csv(op_file)


def ensure_full_year_15min_data(df, year):
  """
  Ensures every meter has every 15-minute interval for the full year.
  Missing intervals are filled with 0.0.
  """
  # 1. Generate every 15-min timestamp for the year (366 days for 2020)
  full_range = pd.date_range(start=f'{year}-01-01 00:00:00', 
                             end=f'{year}-12-31 23:45:00', 
                             freq='15T')
  
  cleaned_list = []
  
  for meter_id in df['meter'].unique():
      meter_df = df[df['meter'] == meter_id].copy()
      
      # Set index to timestamp for reindexing
      meter_df = meter_df.set_index('x_Timestamp')
      
      # Reindex to the master 15-min timeline
      meter_df = meter_df.reindex(full_range)
      
      # 2. Fill Missing Data
      meter_df['meter'] = meter_id
      meter_df['t_kWh'] = meter_df['t_kWh'].fillna(0.0)
      
      # Also zero out electrical parameters if missing
      for col in ['z_Avg Voltage (Volt)', 'z_Avg Current (Amp)', 'y_Freq (Hz)']:
          if col in meter_df.columns:
              meter_df[col] = meter_df[col].fillna(0.0)
      
      # 3. Restore and Re-generate helper columns
      meter_df = meter_df.reset_index().rename(columns={'index': 'x_Timestamp'})
      meter_df['Date'] = meter_df['x_Timestamp'].dt.date
      meter_df['date_without_year'] = meter_df['x_Timestamp'].dt.strftime('%m-%d')
      meter_df['Time'] = meter_df['x_Timestamp'].dt.time
      
      cleaned_list.append(meter_df)
      
  return pd.concat(cleaned_list, ignore_index=True)

#problem with the below plot is it looks like a lot of vertical lines
#plotted on x axis, because consumption keeps going up and down several
#times a day. This visualisation was not great
def plot_timeseries_data_for_given_meters(input_df, meter_list):
  fig, axes = plt.subplots(nrows=len(meter_list), ncols=1, figsize=(15, 20), sharex=True)
  for (m, ax) in zip(meter_list, axes):
    # Isolate data for the specific meter
    met = m['meter'].unique()
    meter_data = m.sort_values('x_Timestamp')
    # Plotting
    ax.plot(meter_data['x_Timestamp'], meter_data['t_kWh'], 
            linewidth=1, label=f'Meter {met}')
    
    # Subplot Styling
    ax.set_ylabel('kWh')
    ax.legend(loc='upper right')
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.set_title(f"Load Profile: {met}", loc='left', fontsize=10)
  plt.xlabel("Full Year Timeline")
  plt.tight_layout()
  plt.show()

#decent visualization. This gives you an idea of how different meters 
#consumption compare with each other
def plot_daily_aggregate_data(meter_list):
  plt.figure(figsize=(15, 8))
  
  for meter in meter_list:
      met = meter.head(1)['meter']
      plt.plot(meter['date_without_year'], meter['t_kWh'], label=f'Meter {met}', alpha=0.8)
  
  plt.title("Total Daily Energy Consumption (Sum of 96 intervals/day)")
  plt.ylabel("Total Daily Energy (kWh)")
  plt.xlabel("Timeline")
  plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
  plt.grid(True, linestyle='--', alpha=0.4)
  plt.tight_layout()
  plt.show()

#decent visualization. This gives you an idea of how different meters 
#consumption compare with each other
def plot_weekly_overlay_data(input_df):
  group = input_df['t_kWh']
  input_df['Lag_week'] = group.shift(7)
  plot_overlay_data(input_df['t_kWh'], input_df['Lag_week'], ['meter without delay', 'meter with delay'])
  
  #plt.title("Total Daily Energy Consumption (Sum of 96 intervals/day)")
  #plt.ylabel("Total Daily Energy (kWh)")
  #plt.xlabel("Timeline")


def plot_overlay_data(ser1, ser2, label_list):
  plt.figure(figsize=(15, 8))
  data_len = len(ser1)
  plt.plot(np.arange(0, data_len), ser1, label=f'{label_list[0]}', alpha=0.8)
  data_len = len(ser2)
  plt.plot(np.arange(0, data_len), ser2, label=f'{label_list[1]}', alpha=0.8)
  plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
  plt.xticks(np.arange(0, data_len, 1)) 
  # Use 'minor' ticks if you want the labels to stay sparse but the grid dense
  plt.gca().set_xticks(np.arange(0, data_len, 1), minor=True)
  plt.grid(True, linestyle='--', alpha=0.4)
  plt.tight_layout()
  plt.show()
  

#decent visualisations. Not sure how to use this though
def plot_candle_plot_of_data(meter_id_list, meter_list):
  for meter_id, meter_data in zip(meter_id_list, meter_list):
    # Filter data for one specific meter
    
    # 2. Aggregating to Daily OHLC
    # This creates the 365 rows for the plot
    daily_ohlc = meter_data.groupby('Date')['t_kWh'].agg(['first', 'max', 'min', 'last'])
    daily_ohlc.columns = ['Open', 'High', 'Low', 'Close']
    
    # 3. Plotting Configuration
    plt.figure(figsize=(20, 8))
    
    # Bullish (Green): Consumption ended higher than it started
    # Bearish (Red): Consumption ended lower than it started
    colors = ['#26a69a' if c >= o else '#ef5350' for o, c in zip(daily_ohlc['Open'], daily_ohlc['Close'])]
    
    # Plot the Wicks (High-Low)
    plt.vlines(daily_ohlc.index, daily_ohlc['Low'], daily_ohlc['High'], color='black', linewidth=0.5)
    
    # Plot the Bodies (Open-Close)
    # Using bar width to fit 365 days clearly
    plt.bar(daily_ohlc.index, daily_ohlc['Close'] - daily_ohlc['Open'], 
            bottom=daily_ohlc['Open'], color=colors, width=0.6, edgecolor='black', linewidth=0.3)
    
    plt.title(f"Annual Energy Signature (Candlestick) - Meter {meter_id}")
    plt.ylabel("Power Consumption (kWh)")
    plt.xlabel("Day of Year (2020)")
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Save/Display the 8 separate figures
    plt.savefig(f"CandlePlot_Meter_{meter_id}.png", dpi=300)
    plt.show()

#most insightful plot
def plot_correlation_between_meters(input_df):
  corr_df = input_df.pivot_table(index='x_Timestamp', columns='meter', values='t_kWh')
  
  # 2. Calculate the Correlation Matrix
  correlation_matrix = corr_df.corr()
  
  # 3. Generate the Heatmap
  plt.figure(figsize=(16, 12))
  sns.heatmap(correlation_matrix, 
              annot=False,       # Set to True if you want to see the numbers
              cmap='RdYlGn',     # Green for high correlation, Red for low
              center=0, 
              linewidths=0.1)
  
  plt.title(f"Inter-Meter Demand Correlation Matrix ({len(correlation_matrix)} Meters)", fontsize=16)
  plt.xlabel("Meter ID")
  plt.ylabel("Meter ID")
  
  plt.tight_layout()
  plt.show()

#very insightful graph, we can see that points in 
#the neigbourhood of identity diagonal have very
#high correlation which we expect to see
def plot_correlation_between_days(input_df):
  # Reshape data so each column is a full day's load profile (96 intervals)
  pivot_df = input_df.pivot(index='Time', columns='Date', values='t_kWh')
  # Calculate the correlation of each day against every other day
  day_corr_matrix = pivot_df.corr()
  # Visualize as a heatmap
  sns.heatmap(day_corr_matrix, cmap='RdYlBu_r')
  plt.show()

def get_meter_summary(input_df):
    # Ensure timestamp is in datetime format
    input_df['x_Timestamp'] = pd.to_datetime(input_df['x_Timestamp'])
    
    # Group by meter and aggregate
    summary = input_df.groupby('meter').agg(
        Start_Date=('x_Timestamp', 'min'),
        End_Date=('x_Timestamp', 'max'),
        Readings_Count=('t_kWh', 'count'),
        Avg_Usage=('t_kWh', 'mean'),
        Peak_Usage=('t_kWh', 'max'),
        Min_Usage=('t_kWh', 'min'),
        Volatility=('t_kWh', 'std')
    ).reset_index()
    
    # Calculate total days covered
    summary['Days_Covered'] = (summary['End_Date'] - summary['Start_Date']).dt.days + 1
    
    return summary

  
def plot_fft_annotated(input_df):
    series = input_df.set_index('x_Timestamp')['t_kWh'].resample('15min').mean().interpolate().values
    
    n = len(series)
    fs = n  # 96 samples per day
    fft_values = np.fft.rfft(series)
    frequencies = np.fft.rfftfreq(n, d=1/fs)
    magnitude = np.abs(fft_values)
    
    # Plotting (excluding index 0 / DC component)
    plt.figure(figsize=(16, 8))
    plt.plot(frequencies[1:], magnitude[1:], color='#3498db', alpha=0.5)
    
    # SMARTER PEAK DETECTION
    # height: only peaks above 10% of max
    # distance: peaks must be at least 10 points apart (avoids jitter)
    peaks, _ = find_peaks(magnitude[1:], height=magnitude[1:].max()*0.1, distance=10)
    
    for p in peaks:
        # +1 because we sliced magnitude from [1:]
        f = frequencies[p+1]
        m = magnitude[p+1]
        
        if f <= 5.0: # Focus on the low frequency range
            plt.scatter(f, m, color='red', s=40, zorder=5)
            plt.annotate(f'{f:.2f}', xy=(f, m), xytext=(0, 10), 
                         textcoords='offset points', ha='center',
                         bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.6))

    plt.xlim(0, 500)
    plt.title("Cleaned FFT Peak Analysis")
    plt.xlabel("Frequency (Cycles per Day)")
    plt.ylabel("Magnitude")
    plt.show()

def generate_harmonic_baseline(series, year=2020):
    """
    Filters 15-min signal to keep only Daily, Semi-Daily, and Weekly harmonics.
    """
    n = len(series)
    # Perform Real FFT
    fft_values = np.fft.rfft(series)
    
    # Calculate the frequency indices we want to keep
    # In a full year of 15-min data (35136 points for 2020):
    daily_idx = 366 if year == 2020 else 365
    weekly_idx = daily_idx // 7
    semi_daily_idx = daily_idx * 2
    
    mask = np.zeros_like(fft_values)
    
    # Keep the 'Average' (Index 0) and the main routine peaks (+/- 2 index buffer)
    significant_indices = [0]
    for base in [daily_idx, semi_daily_idx, weekly_idx]:
        significant_indices.extend(range(base-2, base+3))
    
    for idx in significant_indices:
        if 0 <= idx < len(mask):
            mask[idx] = 1
            
    # Zero out the 'noise' (high frequencies) and transform back
    filtered_fft = fft_values * mask
    return np.fft.irfft(filtered_fft, n=n)


# Conceptual logic for your script using PyWavelets (pywt)
def wavelet_denoising(data):
    # Decompose using Daubechies 4
    coeffs = pywt.wavedec(data, 'db4', level=4)
    
    # Isolation of layers:
    # coeffs[0] is the Seasonal Trend (Low freq)
    # coeffs[1:] are the Spike Layers (High freq)
    
    # Create the 'Clean' Baseline by zeroing out high-freq spikes
    clean_coeffs = [coeffs[0]] + [None] * (len(coeffs) - 1)
    baseline = pywt.waverec(clean_coeffs, 'db4')
    
    return baseline[:len(data)]

def get_meter_df(input_df, meter_id):
  return input_df[input_df["meter"]==meter_id]

def plot_wavelet_mra(series, wavelet='db4', level=4):
  # 1. Decompose the signal
  coeffs = pywt.wavedec(series, wavelet, level=level)
  
  # 2. Plotting
  fig, axes = plt.subplots(level + 2, 1, figsize=(12, 10), sharex=True)
  
  # Original Data
  axes[0].plot(series, color='gray', alpha=0.5)
  axes[0].set_title('Original 15-min Data')
  
  # Approximation (Trend)
  axes[1].plot(coeffs[0], color='blue')
  axes[1].set_ylabel('A4 (Trend)')
  
  # Details (Spikes at different scales)
  for i in range(level):
      axes[i+2].plot(coeffs[i+1], color='red')
      axes[i+2].set_ylabel(f'D{level-i} (Detail)')
      
  plt.tight_layout()
  plt.show()


from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, r2_score

def extract_features(df):
    # Ensure we are dealing with a single meter's continuous timeline
    df = df.copy().sort_values(['x_Timestamp'])
    data = df['t_kWh'].values  # .values is faster than .to_list() for pywt
    
    # 1. WAVELET DECOMPOSITION
    # level=4 matches your visualization in wavelet_decom.png
    coeffs = pywt.wavedec(data, 'db4', level=4)
    
    # 2. FEATURE 1: TREND (The 'Harmonic Baseline' from your plot)[cite: 1]
    trend_coeffs = [coeffs[0]] + [np.zeros_like(c) for c in coeffs[1:]]
    df['wav_trend'] = pywt.waverec(trend_coeffs, 'db4')[:len(data)]
    
    # 3. FEATURE 2: SPIKES (The high-frequency residuals)
    spike_coeffs = [np.zeros_like(coeffs[0])] + coeffs[1:]
    df['wav_spike'] = pywt.waverec(spike_coeffs, 'db4')[:len(data)]
    
    # 4. THE TARGET: What happens 15 minutes from now?
    # We shift -1 so the 'current' row knows the 'next' value
    df['target_kWh'] = df['t_kWh'].shift(-1)
    
    # 5. TIME FEATURES (Optional but highly recommended for AdaBoost)
    df['hour'] = df['x_Timestamp'].dt.hour
    
    return df.dropna(subset=['target_kWh'])

def extract_cyclic_features(df):
    df = df.copy().sort_values(['x_Timestamp'])
    data = df['t_kWh'].values
    
    # 1. Wavelet components (Your core logic)
    coeffs = pywt.wavedec(data, 'db4', level=4)
    trend_coeffs = [coeffs[0]] + [np.zeros_like(c) for c in coeffs[1:]]
    df['wav_trend'] = pywt.waverec(trend_coeffs, 'db4')[:len(data)]
    
    # 2. Velocity (Difference between now and 15 mins ago)
    # This helps the model see if a peak is currently "growing"
    df['velocity'] = df['t_kWh'].diff().fillna(0)
    
    # 3. Acceleration (Change in velocity)
    df['acceleration'] = df['velocity'].diff().fillna(0)

    # 4. Harmonic Time (Capturing the 24-hour cycle)
    # This maps 00:00 to 23:45 onto a circle
    hour_norm = 2 * np.pi * df['x_Timestamp'].dt.hour / 24.0
    df['hour_sin'] = np.sin(hour_norm)
    df['hour_cos'] = np.cos(hour_norm)

    # 5. Target
    df['target_kWh'] = df['t_kWh'].shift(-1)
    
    return df.dropna(subset=['target_kWh'])

def build_prediction_model(train_df, test_df):

    # Prepare data
    train_ready = extract_cyclic_features(train_df)
    test_ready = extract_cyclic_features(test_df)

    features = ['wav_trend', 'wav_spike', 'hour']
    features = ['wav_trend', 'velocity', 'acceleration', 'hour_sin', 'hour_cos', 'day_Sunday','day_Monday','day_Tuesday','day_Wednesday','day_Thursday','day_Friday','day_Saturday',]
    X_train, y_train = train_ready[features], train_ready['target_kWh']
    X_test, y_test = test_ready[features], test_ready['target_kWh']

    # 3. AdaBoost Model
    # It learns from the errors of the previous trees—perfect for tracking energy volatility.
    model = AdaBoostRegressor(
        base_estimator=DecisionTreeRegressor(max_depth=5),
        n_estimators=50,
        learning_rate=0.1
    )
    model.fit(X_train, y_train)
    
    # 4. Predict and Evaluate
    test_ready['pred_kWh'] = model.predict(X_test)
    
    print(f"R2 Score: {r2_score(y_test, test_ready['pred_kWh']):.4f}")
    print(f"MAE: {mean_absolute_error(y_test, test_ready['pred_kWh']):.4f} kWh")
    
    return test_ready


def process_data(ip_file, year):
  op_df = pd.read_csv(ip_file)
  op_df['x_Timestamp'] = pd.to_datetime(op_df['x_Timestamp'])
  op_df = ensure_full_year_15min_data(op_df, year)
  op_df['Date'] = op_df['x_Timestamp'].dt.date
  op_df['date_without_year'] = op_df['x_Timestamp'].dt.strftime('%m-%d')
  op_df['Time'] = pd.to_datetime(op_df['x_Timestamp']).dt.time
  day_dummies = pd.get_dummies(op_df['x_Timestamp'].dt.day_name(), prefix='day')
  op_df = pd.concat([op_df, day_dummies], axis=1)
  return op_df

# Run it on one meter to see the magic
# Main code
#Training data prep

training_df = process_data('Mathura_2020_15t.csv', 2020)
training_meters = training_df['meter'].unique()
training_daily_df = training_df.groupby(['meter', 'date_without_year'])['t_kWh'].sum().reset_index()
print(training_df)


# --- APPLYING TO YOUR DATAFRAME ---
# We apply it per meter to ensure different habits are captured correctly
#training_df['harmonic_baseline'] = training_df.groupby('meter')['t_kWh'].transform(
#    lambda x: generate_harmonic_baseline(x.values, year=2020)
#)

#plot_overlay_data(get_meter_df(training_df, tgt_meter)['t_kWh'], get_meter_df(training_df, tgt_meter)['harmonic_baseline'], ['original', 'harmonic baseline'])
# --- CALCULATE THE SPIKE SIGNAL ---
# This 'Residual' is what your AdaBoost should focus on for anomaly detection
#training_df['spike_signal'] = training_df['t_kWh'] - training_df['harmonic_baseline']
#plot_wavelet_mra(training_df[training_df['meter'] == 'BR04']['t_kWh'])

#Test data prep
test_df = process_data('Mathura_2021_15t.csv', 2021)
tgt_meter = "MH43"
print(test_df)
test_meters = test_df['meter'].unique()
test_daily_df = test_df.groupby(['meter', 'date_without_year'])['t_kWh'].sum().reset_index()

#wav_meter = get_meter_df(training_df, tgt_meter)
#wav = wavelet_denoising(wav_meter['t_kWh'].to_list())
#wav_meter['wavelet'] = wav
#print(wav)
#plot_overlay_data(wav_meter['t_kWh'], wav_meter['wavelet'], ['original', 'harmonic baseline'])

plot_timeseries_data_for_given_meters(training_df, [get_meter_df(training_df, training_meters[1]), get_meter_df(training_df, training_meters[2])])
plot_daily_aggregate_data([get_meter_df(training_daily_df, "BR04"), get_meter_df(test_daily_df, "BR04")])
plot_candle_plot_of_data(["BR02(training)", "BR02(test)"], [get_meter_df(training_df, tgt_meter), get_meter_df(test_df, tgt_meter)])
plot_weekly_overlay_data(get_meter_df(training_daily_df, "BR04"))
#
#
##Aggregate plots
plot_correlation_between_meters(training_df)
plot_correlation_between_days(training_df[training_df['meter']==tgt_meter])
print(get_meter_summary(training_df))
print(get_meter_summary(test_df))
plot_fft_annotated(get_meter_df(training_df, tgt_meter))
plot_fft_annotated(get_meter_df(test_df, tgt_meter))

# Run model
prediction_results = build_prediction_model(get_meter_df(training_df, tgt_meter), get_meter_df(test_df, tgt_meter))

# --- PLOT PREDICTION VS ACTUAL ---
# Isolate a 24-hour window for Meter BR02
sample_plot = prediction_results[prediction_results['meter'] == tgt_meter].iloc[100:196]
plt.figure(figsize=(15, 6))
plt.plot(sample_plot['x_Timestamp'], sample_plot['target_kWh'], label='Actual Next 15m', color='#3498db', marker='o', alpha=0.6)
plt.plot(sample_plot['x_Timestamp'], sample_plot['pred_kWh'], label='AdaBoost Prediction', color='#e67e22', linestyle='--', linewidth=2)
plt.title("Meter BR02: 15-Minute Forward Prediction")
plt.ylabel("Energy (kWh)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

