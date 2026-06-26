import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def generate_synthetic_data(file_path):
    """Generates synthetic pollution source attribution data for development and testing."""
    np.random.seed(42)
    n_samples = 1200
    
    # Generate feature distributions
    traffic_density = np.random.uniform(10, 100, n_samples)
    construction_index = np.random.uniform(5, 50, n_samples)
    industrial_index = np.random.uniform(0, 80, n_samples)
    
    pm25 = (0.6 * traffic_density) + (0.4 * construction_index) + (0.8 * industrial_index) + np.random.normal(15, 5, n_samples)
    pm10 = (0.4 * traffic_density) + (1.2 * construction_index) + (0.9 * industrial_index) + np.random.normal(25, 10, n_samples)
    no2 = (0.9 * traffic_density) + (0.1 * construction_index) + (0.5 * industrial_index) + np.random.normal(10, 3, n_samples)
    
    df = pd.DataFrame({
        'PM25': np.clip(pm25, 5, 500),
        'PM10': np.clip(pm10, 10, 600),
        'NO2': np.clip(no2, 2, 200),
        'Traffic_Density': traffic_density,
        'Construction_Index': construction_index,
        'Industrial_Index': industrial_index
    })
    
    # Rule-based synthetic target assignment with noise
    sources = []
    for idx, row in df.iterrows():
        scores = {
            'Traffic': row['Traffic_Density'] * 1.2 + row['NO2'] * 0.8,
            'Construction': row['Construction_Index'] * 2.0 + (row['PM10'] - row['PM25']) * 0.5,
            'Industry': row['Industrial_Index'] * 1.5 + row['PM25'] * 0.5,
            'Mixed': (row['Traffic_Density'] + row['Construction_Index'] + row['Industrial_Index']) * 0.6
        }
        # Select the source with highest engineered score
        sources.append(max(scores, key=scores.get))
        
    df['Source'] = sources
    
    # Inject a few missing values to validate handling pipelines
    df.loc[np.random.choice(n_samples, 15), 'PM25'] = np.nan
    df.loc[np.random.choice(n_samples, 15), 'Traffic_Density'] = np.nan
    
    df.to_csv(file_path, index=False)
    print(f"[*] Synthetic dataset built successfully at {file_path}")

def main():
    # 1. Pipeline Setup & Directory Verification
    data_dir = 'data'
    model_dir = 'models'
    data_path = os.path.join(data_dir, 'source_data.csv')
    model_path = os.path.join(model_dir, 'source_model.pkl')
    
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    if not os.path.exists(data_path):
        generate_synthetic_data(data_path)
        
    # 2. Load Dataset
    print("[*] Ingesting pollution source data...")
    df = pd.read_csv(data_path)
    
    # 3. Handle Missing Values
    # Impute missing values with column medians to protect against outliers
    feature_cols = ['PM25', 'PM10', 'NO2', 'Traffic_Density', 'Construction_Index', 'Industrial_Index']
    for col in feature_cols:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"[-] Imputed missing values in column '{col}' using median: {median_val:.2f}")

    # 4. Feature-Target Segregation & Split
    X = df[feature_cols]
    y = df['Source']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"[*] Data split into train shape {X_train.shape} and test shape {X_test.shape}")
    
    # 5. Train Random Forest Classifier
    print("[*] Training RandomForestClassifier model...")
    model = RandomForestClassifier(
        n_estimators=150, 
        max_depth=12, 
        random_state=42, 
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # 6. Evaluation Matrix Generation
    predictions = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average='weighted')
    recall = recall_score(y_test, predictions, average='weighted')
    f1 = f1_score(y_test, predictions, average='weighted')
    
    print("\n" + "="*40)
    print("      CLASSIFICATION PERFORMANCE      ")
    print("="*40)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f} (Weighted)")
    print(f"Recall:    {recall:.4f} (Weighted)")
    print(f"F1 Score:  {f1:.4f} (Weighted)")
    print("="*40)
    
    # 7. Extract Feature Importances
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    print("\nFeature Ranking by Variance Explanation:")
    for f in range(X.shape[1]):
        print(f"{f + 1}. {X.columns[indices[f]]:<20} : {importances[indices[f]]:.4f}")
    print("="*40 + "\n")
    
    # 8. Serialize and Persist Model Architecture
    print(f"[*] Exporting serialized model binary to {model_path}...")
    joblib.dump(model, model_path)
    print("[*] Source attribution training pipeline completed successfully.")

if __name__ == '__main__':
    main()