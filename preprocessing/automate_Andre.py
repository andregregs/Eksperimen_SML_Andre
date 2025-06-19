"""
Automate Preprocessing for Heart Disease Dataset
Author: Andre
Date: June 2025
"""

import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

def load_data(file_path='../heart.csv'):
    """
    Load dataset from CSV file
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded dataset
    """
    try:
        df = pd.read_csv(file_path)
        print(f"✅ Data loaded successfully! Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None

def create_preprocessor():
    """
    Create preprocessing pipeline
    
    Returns:
        ColumnTransformer: Preprocessing pipeline
    """
    # Define feature types
    numeric_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
    
    # Create transformers
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combine transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    print("✅ Preprocessor created successfully!")
    return preprocessor

def split_data(df, test_size=0.2, random_state=42):
    """
    Split data into features and target, then train-test split
    
    Args:
        df (pd.DataFrame): Input dataset
        test_size (float): Test set ratio
        random_state (int): Random seed
        
    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    # Separate features and target
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"✅ Data split completed!")
    print(f"   Training set: {X_train.shape}")
    print(f"   Test set: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test

def preprocess_data(X_train, X_test, preprocessor):
    """
    Apply preprocessing to training and test sets
    
    Args:
        X_train: Training features
        X_test: Test features
        preprocessor: Fitted preprocessor
        
    Returns:
        tuple: Processed X_train, X_test, feature_names
    """
    # Fit and transform training data
    X_train_processed = preprocessor.fit_transform(X_train)
    
    # Transform test data
    X_test_processed = preprocessor.transform(X_test)
    
    # Get feature names
    numeric_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
    
    feature_names = numeric_features.copy()
    if hasattr(preprocessor.named_transformers_['cat']['onehot'], 'get_feature_names_out'):
        cat_feature_names = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_features)
        feature_names.extend(cat_feature_names)
    
    print(f"✅ Data preprocessing completed!")
    print(f"   Features: {len(feature_names)}")
    
    return X_train_processed, X_test_processed, feature_names

def save_processed_data(X_train_processed, X_test_processed, y_train, y_test, 
                       feature_names, preprocessor, output_dir='data/processed'):
    """
    Save processed data to files
    
    Args:
        X_train_processed: Processed training features
        X_test_processed: Processed test features
        y_train: Training target
        y_test: Test target
        feature_names: List of feature names
        preprocessor: Fitted preprocessor
        output_dir: Output directory
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Convert to DataFrame
    X_train_df = pd.DataFrame(X_train_processed, columns=feature_names)
    X_test_df = pd.DataFrame(X_test_processed, columns=feature_names)
    
    # Save datasets
    X_train_df.to_csv(f'{output_dir}/X_train.csv', index=False)
    X_test_df.to_csv(f'{output_dir}/X_test.csv', index=False)
    pd.Series(y_train).to_csv(f'{output_dir}/y_train.csv', index=False, header=['target'])
    pd.Series(y_test).to_csv(f'{output_dir}/y_test.csv', index=False, header=['target'])
    
    # Save complete processed dataset
    heart_processed = pd.concat([X_train_df, X_test_df], axis=0)
    heart_processed['target'] = pd.concat([pd.Series(y_train), pd.Series(y_test)], axis=0)
    heart_processed.to_csv(f'{output_dir}/heart_processed.csv', index=False)
    
    # Save preprocessor and feature names
    joblib.dump(preprocessor, 'models/preprocessor.pkl')
    joblib.dump(feature_names, 'models/feature_names.pkl')
    
    print(f"✅ All files saved successfully in '{output_dir}' and 'models'!")
    return True

def run_full_preprocessing(input_file='heart.csv', output_dir='data/processed'):
    """
    Run complete preprocessing pipeline
    
    Args:
        input_file (str): Path to input CSV file
        output_dir (str): Output directory for processed data
        
    Returns:
        bool: Success status
    """
    print("🚀 Starting Heart Disease Data Preprocessing Pipeline...")
    print("="*60)
    
    try:
        # Step 1: Load data
        df = load_data(input_file)
        if df is None:
            return False
        
        # Step 2: Create preprocessor
        preprocessor = create_preprocessor()
        
        # Step 3: Split data
        X_train, X_test, y_train, y_test = split_data(df)
        
        # Step 4: Preprocess data
        X_train_processed, X_test_processed, feature_names = preprocess_data(
            X_train, X_test, preprocessor
        )
        
        # Step 5: Save processed data
        save_processed_data(
            X_train_processed, X_test_processed, y_train, y_test,
            feature_names, preprocessor, output_dir
        )
        
        print("="*60)
        print("🎉 PREPROCESSING PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"📁 Output files saved in: {output_dir}")
        print(f"🔧 Models saved in: models/")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in preprocessing pipeline: {e}")
        return False

if __name__ == "__main__":
    # Run preprocessing when script is executed directly
    success = run_full_preprocessing()
    if success:
        print("\n✅ Ready for model training!")
    else:
        print("\n❌ Preprocessing failed!")