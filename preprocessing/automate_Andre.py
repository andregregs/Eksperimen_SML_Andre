import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import os
import warnings
warnings.filterwarnings('ignore')

def load_data(file_path='heart.csv'):
    """
    Load dataset from CSV file
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded dataset
    """
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    print(f"Data loaded successfully! Shape: {df.shape}")
    return df

def prepare_features(df):
    """
    Separate features and target variable
    
    Args:
        df (pd.DataFrame): Input dataset
        
    Returns:
        tuple: (X, y, numeric_features, categorical_features)
    """
    print("Preparing features and target...")
    
    # Separate features and target
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Define feature types
    numeric_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
    
    print(f"Features prepared: {len(numeric_features)} numeric, {len(categorical_features)} categorical")
    return X, y, numeric_features, categorical_features

def create_preprocessor(numeric_features, categorical_features):
    """
    Create preprocessing pipeline
    
    Args:
        numeric_features (list): List of numeric feature names
        categorical_features (list): List of categorical feature names
        
    Returns:
        ColumnTransformer: Preprocessing pipeline
    """
    print("Creating preprocessing pipeline...")
    
    # Numeric preprocessing
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Categorical preprocessing
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    print("Preprocessing pipeline created successfully!")
    return preprocessor

def split_and_process_data(X, y, test_size=0.2, random_state=42):
    """
    Split data and apply preprocessing
    
    Args:
        X (pd.DataFrame): Features
        y (pd.Series): Target variable
        test_size (float): Test set size ratio
        random_state (int): Random state for reproducibility
        
    Returns:
        tuple: (X_train_processed, X_test_processed, y_train, y_test, feature_names)
    """
    print("Splitting data...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Data split: Train {X_train.shape}, Test {X_test.shape}")
    
    # Get feature names for preprocessing
    numeric_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
    
    # Create and fit preprocessor
    preprocessor = create_preprocessor(numeric_features, categorical_features)
    
    print("Applying preprocessing...")
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Get feature names after preprocessing
    feature_names = []
    feature_names.extend(numeric_features)
    
    if hasattr(preprocessor.named_transformers_['cat']['onehot'], 'get_feature_names_out'):
        cat_feature_names = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_features)
        feature_names.extend(cat_feature_names)
    
    print(f"Preprocessing completed! Features: {len(feature_names)}")
    return X_train_processed, X_test_processed, y_train, y_test, feature_names

def save_processed_data(X_train_processed, X_test_processed, y_train, y_test, feature_names, output_dir='preprocessing/heart_preprocessing'):
    """
    Save processed data to CSV files
    
    Args:
        X_train_processed: Processed training features
        X_test_processed: Processed test features
        y_train: Training target
        y_test: Test target
        feature_names: List of feature names
        output_dir: Output directory path
    """
    print(f"Saving processed data to {output_dir}...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
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
    
    # Save feature names
    pd.DataFrame({'feature_names': feature_names}).to_csv(f'{output_dir}/feature_names.csv', index=False)
    
    print("✅ All files saved successfully!")
    print(f"- X_train.csv: {X_train_df.shape}")
    print(f"- X_test.csv: {X_test_df.shape}")
    print(f"- y_train.csv: {len(y_train)} samples")
    print(f"- y_test.csv: {len(y_test)} samples")
    print(f"- heart_processed.csv: {heart_processed.shape}")
    print(f"- feature_names.csv: {len(feature_names)} features")

def main_preprocessing_pipeline(input_file='heart.csv', output_dir='preprocessing/heart_preprocessing'):
    """
    Main function to run the complete preprocessing pipeline
    
    Args:
        input_file (str): Path to input CSV file
        output_dir (str): Output directory for processed files
        
    Returns:
        dict: Summary of processed data
    """
    print("="*60)
    print("AUTOMATED HEART DISEASE PREPROCESSING PIPELINE")
    print("="*60)
    
    try:
        # Step 1: Load data
        df = load_data(input_file)
        
        # Step 2: Prepare features
        X, y, numeric_features, categorical_features = prepare_features(df)
        
        # Step 3: Split and process data
        X_train_processed, X_test_processed, y_train, y_test, feature_names = split_and_process_data(X, y)
        
        # Step 4: Save processed data
        save_processed_data(X_train_processed, X_test_processed, y_train, y_test, feature_names, output_dir)
        
        # Return summary
        summary = {
            'input_shape': df.shape,
            'train_shape': X_train_processed.shape,
            'test_shape': X_test_processed.shape,
            'n_features': len(feature_names),
            'output_dir': output_dir
        }
        
        print("\n🎉 PREPROCESSING PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"Summary: {summary}")
        
        return summary
        
    except Exception as e:
        print(f"❌ Error in preprocessing pipeline: {str(e)}")
        raise e

if __name__ == "__main__":
    # Run the main preprocessing pipeline
    result = main_preprocessing_pipeline()
    print("\nPreprocessing completed. Data ready for model training!")