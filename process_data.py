import pandas as pd
import sys
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


convert_to_15min_data(sys.argv[1], sys.argv[2])
