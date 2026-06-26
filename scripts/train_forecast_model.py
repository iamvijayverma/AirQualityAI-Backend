import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def create_dummy_data(file_path):
    """Generates synthetic AQI data if the CSV is missing, ensuring the script is runnable."""
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=1000, freq='h') # hourly data
    
    # Simulate a daily AQI cycle with some noise
    base_aqi = 100 + np.sin(np.linspace(0, 50, 1000)) * 50
    noise = np.random.normal(0, 10, 1000)
    aqi = np.clip(base_aqi + noise, 0, 500) 
    
    df = pd.DataFrame({'timestamp': dates, 'AQI': aqi})
    
    # Introduce random missing values to test data cleaning
    df.loc[10:15, 'AQI'] = np.nan
    df.loc[500:505, 'AQI'] = np.nan
    
    df.to_csv(file_path, index=False)
    print(f"[*] Created synthetic dataset at {file_path} for testing.")

def main():
    # 1. Setup Directories & Paths
    data_dir = 'data'
    model_dir = 'models'
    data_path = os.path.join(data_dir, 'aqi_data.csv')
    model_path = os.path.join(model_dir, 'aqi_model.pkl')
    
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    if not os.path.exists(data_path):
        create_dummy_data(data_path)
        
    # 2. Load AQI Data
    print("[*] Loading data...")
    df = pd.read_csv(data_path)
    
    # 3. Perform Data Cleaning & Handle Missing Values
    print("[*] Cleaning data...")
    # Time-series data is best cleaned with forward fill, falling back to backward fill
    df['AQI'] = df['AQI'].ffill().bfill()
    
    # 4. Feature Engineering (Lags)
    print("[*] Creating lag features...")
    df['AQI_1'] = df['AQI'].shift(1)
    df['AQI_6'] = df['AQI'].shift(6)
    df['AQI_24'] = df['AQI'].shift(24)
    
    # Drop rows with NaN values created by the shifting process
    df = df.dropna()
    
    # Define features (X) and target (y)
    X = df[['AQI_1', 'AQI_6', 'AQI_24']]
    y = df['AQI']
    
    # 5. Train-Test Split
    print("[*] Splitting dataset...")
    # shuffle=False is critical for time-series to prevent data leakage from future to past
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # 6. Train RandomForestRegressor
    print("[*] Training RandomForestRegressor...")
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # 7. Predict AQI
    print("[*] Generating predictions...")
    predictions = model.predict(X_test)
    
    # 8. Calculate Metrics
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    
    print("\n==================================")
    print("      MODEL EVALUATION METRICS      ")
    print("==================================")
    print(f"MAE  (Mean Absolute Error): {mae:.4f}")
    print(f"RMSE (Root Mean Sq Error):  {rmse:.4f}")
    print(f"R²   (Coefficient of Det.): {r2:.4f}")
    print("==================================\n")
    
    # 9. Save the Model
    print(f"[*] Saving trained model to {model_path}...")
    joblib.dump(model, model_path)
    print("[*] Pipeline execution completed successfully.")

if __name__ == "__main__":
    main()