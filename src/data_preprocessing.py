import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(filepath):
    """
    Load housing data from CSV file.
    
    Args:
        filepath (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded data
    """
    try:
        data = pd.read_csv(filepath)
        print(f"Data loaded successfully. Shape: {data.shape}")
        return data
    except FileNotFoundError:
        print(f"Error: File {filepath} not found.")
        return None

def explore_data(data):
    """
    Explore and display basic statistics of the data.
    
    Args:
        data (pd.DataFrame): Input data
    """
    print("\n=== Data Overview ===")
    print(f"Shape: {data.shape}")
    print(f"\nColumn Names and Types:\n{data.dtypes}")
    print(f"\nFirst few rows:\n{data.head()}")
    print(f"\nBasic Statistics:\n{data.describe()}")
    print(f"\nMissing Values:\n{data.isnull().sum()}")

def handle_missing_values(data):
    """
    Handle missing values in the dataset.
    
    Args:
        data (pd.DataFrame): Input data
        
    Returns:
        pd.DataFrame: Data with missing values handled
    """
    # Drop rows with missing values (can also use fillna for other strategies)
    data_cleaned = data.dropna()
    print(f"Rows after removing missing values: {len(data_cleaned)}")
    return data_cleaned

def prepare_features_target(data, feature_columns, target_column):
    """
    Prepare features (X) and target (y) from the data.
    
    Args:
        data (pd.DataFrame): Input data
        feature_columns (list): Names of feature columns
        target_column (str): Name of target column
        
    Returns:
        tuple: (X, y) - Features and target
    """
    X = data[feature_columns]
    y = data[target_column]
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    return X, y

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets.
    
    Args:
        X (pd.DataFrame): Features
        y (pd.Series): Target variable
        test_size (float): Proportion of test set
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"\nTraining set size: {len(X_train)}")
    print(f"Testing set size: {len(X_test)}")
    return X_train, X_test, y_train, y_test

def scale_features(X_train, X_test):
    """
    Standardize features using StandardScaler.
    
    Args:
        X_train (pd.DataFrame): Training features
        X_test (pd.DataFrame): Testing features
        
    Returns:
        tuple: (X_train_scaled, X_test_scaled)
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("Features scaled successfully.")
    return X_train_scaled, X_test_scaled, scaler
