import pandas as pd
import numpy as np
import sys

def clean_smart_meter_data(input_file, output_file):
    # 1. Load Data
    df = pd.read_csv(input_file)
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])

    # 2. Chronological Sorting (CRITICAL)
    meta = ['CONS_NO', 'FLAG']
    dates = [c for c in df.columns if c not in meta]
    # Sort dates so the time-series flow is correct: Jan -> Feb -> March...
    sorted_dates = sorted(dates, key=lambda x: pd.to_datetime(x))
    df = df[meta + sorted_dates]

    # 3. Create "Missingness" Features (The 'Signal' from your image)
    # This helps the Unsupervised model know what was originally missing
    df['missing_count'] = df[sorted_dates].isnull().sum(axis=1)
    df['missing_ratio'] = df['missing_count'] / len(sorted_dates)

    # 4. Contextual Filling
    # Instead of just 0, we use a forward-fill (FFill) limited to 3 days.
    # If data is missing for >3 days, we set it to 0 (potential tampering/cutoff).
    consumption = df[sorted_dates]
    
    # Apply forward fill to handle small communication gaps (Sensor Noise)
    consumption = consumption.ffill(axis=1, limit=3)
    
    # Remaining NaNs are long gaps (Potential Outages or Theft) -> Fill with 0
    consumption = consumption.fillna(0)

    # 5. Final Assembly
    df_final = pd.concat([df[meta + ['missing_ratio']], consumption], axis=1)
    
    # 6. Save
    df_final.to_csv(output_file, index=False)
    print(f"Data ready for ML. Saved to {output_file}")
    return df_final

# Execute
df_ready = clean_smart_meter_data(sys.argv[1], sys.argv[2])
