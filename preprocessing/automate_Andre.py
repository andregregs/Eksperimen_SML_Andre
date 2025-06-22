"""
Automated Preprocessing for Heart Disease Dataset
Author: Andre
Description: Script untuk otomatisasi preprocessing data heart disease
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

def load_data(file_path):
    """
    Load dataset dari file CSV
    
    Args:
        file_path (str): Path ke file CSV
        
    Returns:
        pd.DataFrame: Dataset yang sudah dimuat
    """
    try:
        df = pd.read_csv(file_path)
        print(f"✅ Dataset berhasil dimuat dengan shape: {df.shape}")
        return df
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None

def create_preprocessor():
    """
    Membuat preprocessing pipeline
    
    Returns:
        ColumnTransformer: Preprocessor yang sudah dikonfigurasi
    """
    # Define feature types
    numeric_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
    
    # Create preprocessing pipelines
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combine preprocessing pipelines
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return preprocessor

def preprocess_data(df, test_size=0.2, random_state=42):
    """
    Melakukan preprocessing lengkap pada dataset
    
    Args:
        df (pd.DataFrame): Dataset input
        test_size (float): Proporsi data test
        random_state (int): Random state untuk reproducibility
        
    Returns:
        tuple: (X_train_processed, X_test_processed, y_train, y_test, preprocessor)
    """
    # Pisahkan fitur dan target
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Create and fit preprocessor
    preprocessor = create_preprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    print(f"✅ Preprocessing selesai!")
    print(f"   - X_train shape: {X_train_processed.shape}")
    print(f"   - X_test shape: {X_test_processed.shape}")
    print(f"   - Training target distribution: {pd.Series(y_train).value_counts().to_dict()}")
    print(f"   - Testing target distribution: {pd.Series(y_test).value_counts().to_dict()}")
    
    return X_train_processed, X_test_processed, y_train, y_test, preprocessor

def save_processed_data(X_train, X_test, y_train, y_test, preprocessor, output_dir='data/processed'):
    """
    Menyimpan data yang sudah diproses
    
    Args:
        X_train: Training features
        X_test: Testing features  
        y_train: Training target
        y_test: Testing target
        preprocessor: Fitted preprocessor
        output_dir (str): Directory output
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save processed data
    pd.DataFrame(X_train).to_csv(f'{output_dir}/X_train.csv', index=False)
    pd.DataFrame(X_test).to_csv(f'{output_dir}/X_test.csv', index=False)
    pd.Series(y_train).to_csv(f'{output_dir}/y_train.csv', index=False, header=['target'])
    pd.Series(y_test).to_csv(f'{output_dir}/y_test.csv', index=False, header=['target'])
    
    # Save preprocessor
    joblib.dump(preprocessor, f'{output_dir}/preprocessor.pkl')
    
    print(f"✅ Data tersimpan di: {output_dir}/")

def main(input_file='../heart.csv', output_dir='data/processed'):
    """
    Fungsi utama untuk menjalankan seluruh pipeline preprocessing
    
    Args:
        input_file (str): Path ke file dataset
        output_dir (str): Directory untuk menyimpan hasil
    """
    print("="*50)
    print("AUTOMATED HEART DISEASE PREPROCESSING")
    print("="*50)
    
    # Load data
    df = load_data(input_file)
    if df is None:
        return False
    
    # Basic data info
    print(f"\nDataset Info:")
    print(f"- Shape: {df.shape}")
    print(f"- Missing values: {df.isnull().sum().sum()}")
    print(f"- Target distribution: {df['target'].value_counts().to_dict()}")
    
    # Preprocess data
    print(f"\nMemulai preprocessing...")
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df)
    
    # Save processed data
    print(f"\nMenyimpan hasil preprocessing...")
    save_processed_data(X_train, X_test, y_train, y_test, preprocessor, output_dir)
    
    print("\n" + "="*50)
    print("PREPROCESSING SELESAI!")
    print("="*50)
    return True

if __name__ == "__main__":
    # Jalankan preprocessing otomatis
    success = main()
    if success:
        print("✅ Semua proses berhasil!")
    else:
        print("❌ Terjadi error dalam proses!")