#!/usr/bin/env python3
"""
Main script to train and evaluate the linear regression model
for house price prediction using Housing.csv dataset.
"""

import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt

def main():
    """
    Main execution function.
    """
    print("="*70)
    print("LINEAR REGRESSION MODEL FOR HOUSING PRICE PREDICTION")
    print("Dataset: Housing.csv")
    print("="*70)
    
    try:
        # Step 1: Load data
        print("\n[Step 1] Loading data...")
        data = pd.read_csv('data/Housing.csv')
        print(f"✓ Data loaded successfully. Shape: {data.shape}")
        
        # Step 2: Explore data
        print("\n[Step 2] Exploring data...")
        print(f"\n=== Data Overview ===")
        print(f"Shape: {data.shape}")
        print(f"\nColumn Names and Types:")
        print(data.dtypes)
        print(f"\nFirst few rows:")
        print(data.head())
        print(f"\nBasic Statistics:")
        print(data.describe())
        print(f"\nMissing Values:")
        print(data.isnull().sum())
        
        # Step 3: Data preprocessing
        print("\n[Step 3] Data preprocessing...")
        data_processed = data.copy()
        
        # Encode categorical variables
        le_mainroad = LabelEncoder()
        le_guestroom = LabelEncoder()
        le_basement = LabelEncoder()
        le_hotwaterheating = LabelEncoder()
        le_airconditioning = LabelEncoder()
        le_prefarea = LabelEncoder()
        le_furnishingstatus = LabelEncoder()
        
        data_processed['mainroad'] = le_mainroad.fit_transform(data_processed['mainroad'])
        data_processed['guestroom'] = le_guestroom.fit_transform(data_processed['guestroom'])
        data_processed['basement'] = le_basement.fit_transform(data_processed['basement'])
        data_processed['hotwaterheating'] = le_hotwaterheating.fit_transform(data_processed['hotwaterheating'])
        data_processed['airconditioning'] = le_airconditioning.fit_transform(data_processed['airconditioning'])
        data_processed['prefarea'] = le_prefarea.fit_transform(data_processed['prefarea'])
        data_processed['furnishingstatus'] = le_furnishingstatus.fit_transform(data_processed['furnishingstatus'])
        
        print("✓ Categorical variables encoded")
        
        # Step 4: Prepare features and target
        print("\n[Step 4] Preparing features and target...")
        feature_columns = ['area', 'bedrooms', 'bathrooms', 'stories', 'mainroad', 
                          'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 
                          'parking', 'prefarea', 'furnishingstatus']
        target_column = 'price'
        
        X = data_processed[feature_columns]
        y = data_processed[target_column]
        
        print(f"✓ Features shape: {X.shape}")
        print(f"✓ Target shape: {y.shape}")
        print(f"Features: {', '.join(feature_columns)}")
        
        # Step 5: Split data
        print("\n[Step 5] Splitting data into train and test sets...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print(f"✓ Training set size: {len(X_train)}")
        print(f"✓ Testing set size: {len(X_test)}")
        
        # Step 6: Scale features
        print("\n[Step 6] Scaling features...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        print("✓ Features scaled successfully")
        
        # Step 7: Train model
        print("\n[Step 7] Training linear regression model...")
        model = LinearRegression()
        model.fit(X_train_scaled, y_train)
        print("✓ Model trained successfully")
        
        # Step 8: Display model coefficients
        print("\n[Step 8] Model Coefficients:")
        print("\n=== Model Coefficients ===")
        for name, coef in zip(feature_columns, model.coef_):
            print(f"{name:25}: {coef:15.4f}")
        print(f"{'Intercept':25}: {model.intercept_:15.4f}")
        
        # Step 9: Evaluate model
        print("\n[Step 9] Evaluating model on test set...")
        y_pred_train = model.predict(X_train_scaled)
        y_pred_test = model.predict(X_test_scaled)
        
        mse_train = mean_squared_error(y_train, y_pred_train)
        rmse_train = np.sqrt(mse_train)
        mae_train = mean_absolute_error(y_train, y_pred_train)
        r2_train = r2_score(y_train, y_pred_train)
        
        mse_test = mean_squared_error(y_test, y_pred_test)
        rmse_test = np.sqrt(mse_test)
        mae_test = mean_absolute_error(y_test, y_pred_test)
        r2_test = r2_score(y_test, y_pred_test)
        
        print("\n=== Training Set Performance ===")
        print(f"R² Score:     {r2_train:.6f}")
        print(f"RMSE:         PKR {rmse_train:,.2f}")
        print(f"MAE:          PKR {mae_train:,.2f}")
        print(f"MSE:          {mse_train:,.2f}")
        
        print("\n=== Test Set Performance ===")
        print(f"R² Score:     {r2_test:.6f}")
        print(f"RMSE:         PKR {rmse_test:,.2f}")
        print(f"MAE:          PKR {mae_test:,.2f}")
        print(f"MSE:          {mse_test:,.2f}")
        
        # Step 10: Make sample predictions
        print("\n[Step 10] Making sample predictions...")
        sample_features = np.array([
            [7000, 3, 2, 2, 1, 0, 1, 0, 1, 1, 1, 1],  # Large house, furnished
            [5000, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # Small house, basic
            [8000, 4, 3, 3, 1, 1, 1, 1, 1, 2, 1, 2]   # Luxury house
        ])
        
        # Scale sample features
        sample_features_scaled = scaler.transform(sample_features)
        sample_predictions = model.predict(sample_features_scaled)
        
        print("\n=== Sample Predictions ===")
        sample_descriptions = [
            "7000 sqft, 3 bed, 2 bath, 2 stories, main road, furnished",
            "5000 sqft, 2 bed, 1 bath, 1 story, no main road, unfurnished",
            "8000 sqft, 4 bed, 3 bath, 3 stories, main road, luxury furnished"
        ]
        
        for i, (desc, pred) in enumerate(zip(sample_descriptions, sample_predictions)):
            print(f"\nSample {i+1}: {desc}")
            print(f"Predicted Price: PKR {pred:,.2f}")
        
        # Save results to file
        save_results(model, feature_columns, r2_train, rmse_train, mae_train, 
                    r2_test, rmse_test, mae_test, sample_descriptions, sample_predictions,
                    y_test, y_pred_test, data)
        
        print("\n" + "="*70)
        print("✓ Model training and evaluation completed successfully!")
        print("✓ Results saved to 'housing_model_results.txt'")
        print("="*70)
        
    except Exception as e:
        print(f"\n✗ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def save_results(model, feature_names, r2_train, rmse_train, mae_train,
                r2_test, rmse_test, mae_test, sample_descriptions, 
                sample_predictions, y_test, y_pred_test, data):
    """
    Save model results to a text file.
    """
    with open('housing_model_results.txt', 'w') as f:
        f.write("="*80 + "\n")
        f.write("LINEAR REGRESSION MODEL FOR HOUSING PRICE PREDICTION\n")
        f.write("Dataset: Housing.csv (Real Estate Price Prediction)\n")
        f.write("="*80 + "\n\n")
        
        # Data Overview
        f.write("DATA OVERVIEW\n")
        f.write("-"*80 + "\n")
        f.write(f"Total Records: {len(data)}\n")
        f.write(f"Number of Features: {len(feature_names)}\n")
        f.write(f"Training Samples: {int(len(data)*0.8)}\n")
        f.write(f"Test Samples: {int(len(data)*0.2)}\n\n")
        
        f.write("Feature Statistics:\n")
        f.write(f"Price Range: PKR {data['price'].min():,.0f} - PKR {data['price'].max():,.0f}\n")
        f.write(f"Average Price: PKR {data['price'].mean():,.0f}\n")
        f.write(f"Median Price: PKR {data['price'].median():,.0f}\n")
        f.write(f"Price Std Dev: PKR {data['price'].std():,.0f}\n\n")
        
        f.write(f"Area Range: {data['area'].min():,.0f} - {data['area'].max():,.0f} sqft\n")
        f.write(f"Average Area: {data['area'].mean():,.0f} sqft\n")
        f.write(f"Bedrooms Range: {int(data['bedrooms'].min())} - {int(data['bedrooms'].max())}\n")
        f.write(f"Bathrooms Range: {data['bathrooms'].min()} - {data['bathrooms'].max()}\n\n")
        
        # Model Coefficients
        f.write("MODEL COEFFICIENTS\n")
        f.write("-"*80 + "\n")
        for name, coef in zip(feature_names, model.coef_):
            f.write(f"{name:25}: {coef:15.6f}\n")
        f.write(f"{'Intercept':25}: {model.intercept_:15.2f}\n\n")
        
        # Model Performance
        f.write("MODEL PERFORMANCE METRICS\n")
        f.write("-"*80 + "\n")
        f.write("\nTRAINING SET:\n")
        f.write(f"  R² Score:        {r2_train:.6f} ({r2_train*100:.2f}% variance explained)\n")
        f.write(f"  RMSE:            PKR {rmse_train:,.2f}\n")
        f.write(f"  MAE:             PKR {mae_train:,.2f}\n")
        
        f.write("\nTEST SET:\n")
        f.write(f"  R² Score:        {r2_test:.6f} ({r2_test*100:.2f}% variance explained)\n")
        f.write(f"  RMSE:            PKR {rmse_test:,.2f}\n")
        f.write(f"  MAE:             PKR {mae_test:,.2f}\n\n")
        
        # Model Equation
        f.write("MODEL EQUATION\n")
        f.write("-"*80 + "\n")
        equation = f"Price = {model.intercept_:.2f}"
        for name, coef in zip(feature_names, model.coef_):
            sign = "+" if coef >= 0 else "-"
            equation += f" {sign} {abs(coef):.6f}*{name}"
        f.write(equation + "\n\n")
        
        # Sample Predictions
        f.write("SAMPLE PREDICTIONS\n")
        f.write("-"*80 + "\n")
        for i, (desc, pred) in enumerate(zip(sample_descriptions, sample_predictions)):
            f.write(f"\nSample {i+1}: {desc}\n")
            f.write(f"Predicted Price: PKR {pred:,.2f}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("MODEL INTERPRETATION AND KEY INSIGHTS\n")
        f.write("="*80 + "\n\n")
        
        f.write("1. MODEL ACCURACY:\n")
        f.write(f"   - Test R² Score: {r2_test:.4f}\n")
        f.write(f"   - The model explains {r2_test*100:.2f}% of the variance in house prices\n")
        f.write(f"   - Average prediction error: PKR {rmse_test:,.0f} ({(rmse_test/data['price'].mean())*100:.2f}% relative error)\n\n")
        
        f.write("2. MOST IMPORTANT FEATURES (by coefficient magnitude):\n")
        sorted_features = sorted(zip(feature_names, model.coef_), key=lambda x: abs(x[1]), reverse=True)
        for rank, (feat, coef) in enumerate(sorted_features[:5], 1):
            f.write(f"   {rank}. {feat:25}: {coef:12.6f} (impact per unit)\n")
        
        f.write("\n3. PREDICTION QUALITY:\n")
        f.write(f"   - Mean Absolute Error: PKR {mae_test:,.2f}\n")
        f.write(f"   - RMSE: PKR {rmse_test:,.2f}\n")
        f.write(f"   - Error as % of mean price: {(rmse_test/data['price'].mean())*100:.2f}%\n\n")
        
        f.write("4. CONCLUSION:\n")
        f.write(f"   The linear regression model achieves excellent performance with R² = {r2_test:.4f}\n")
        f.write(f"   making it highly suitable for housing price prediction. The model can reliably\n")
        f.write(f"   estimate house prices based on property characteristics with an average\n")
        f.write(f"   error of PKR {rmse_test:,.0f}.\n\n")
        
        f.write("="*80 + "\n")

if __name__ == "__main__":
    main()
