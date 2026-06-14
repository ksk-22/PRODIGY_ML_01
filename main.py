#!/usr/bin/env python3
"""
Main script to train and evaluate the linear regression model
for house price prediction.
"""

import sys
import pandas as pd
import numpy as np
from src.data_preprocessing import (
    load_data, explore_data, handle_missing_values,
    prepare_features_target, split_data, scale_features
)
from src.model import HousePricePredictor
from src.visualization import (
    plot_actual_vs_predicted, plot_residuals,
    plot_feature_importance, plot_data_distribution
)

def main():
    """
    Main execution function.
    """
    print("="*60)
    print("Linear Regression Model for House Price Prediction")
    print("="*60)
    
    # Configuration
    DATA_FILE = 'data/housing_data.csv'
    FEATURE_COLUMNS = ['square_footage', 'bedrooms', 'bathrooms']
    TARGET_COLUMN = 'price'
    
    try:
        # Step 1: Load data
        print("\n[Step 1] Loading data...")
        data = load_data(DATA_FILE)
        if data is None:
            print("Error: Could not load data. Please ensure the CSV file exists.")
            return
        
        # Step 2: Explore data
        print("\n[Step 2] Exploring data...")
        explore_data(data)
        
        # Step 3: Handle missing values
        print("\n[Step 3] Handling missing values...")
        data = handle_missing_values(data)
        
        # Step 4: Prepare features and target
        print("\n[Step 4] Preparing features and target...")
        X, y = prepare_features_target(data, FEATURE_COLUMNS, TARGET_COLUMN)
        
        # Step 5: Visualize data distribution
        print("\n[Step 5] Visualizing data distribution...")
        plot_data_distribution(data, FEATURE_COLUMNS, TARGET_COLUMN)
        
        # Step 6: Split data
        print("\n[Step 6] Splitting data into train and test sets...")
        X_train, X_test, y_train, y_test = split_data(X, y)
        
        # Step 7: Scale features (optional but recommended)
        print("\n[Step 7] Scaling features...")
        X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
        
        # Step 8: Train model
        print("\n[Step 8] Training linear regression model...")
        model = HousePricePredictor()
        model.train(X_train_scaled, y_train)
        
        # Step 9: Display model coefficients
        print("\n[Step 9] Model coefficients:")
        model.display_coefficients(FEATURE_COLUMNS)
        
        # Step 10: Evaluate model
        print("\n[Step 10] Evaluating model on test set...")
        metrics, y_pred = model.evaluate(X_test_scaled, y_test)
        model.display_metrics(metrics)
        
        # Step 11: Visualize results
        print("\n[Step 11] Generating visualizations...")
        plot_actual_vs_predicted(y_test, y_pred)
        plot_residuals(y_test, y_pred)
        plot_feature_importance(FEATURE_COLUMNS, model.coefficients)
        
        # Step 12: Make sample predictions
        print("\n[Step 12] Making sample predictions...")
        sample_features = np.array([
            [2000, 3, 2],  # 2000 sqft, 3 bedrooms, 2 bathrooms
            [1500, 2, 1],  # 1500 sqft, 2 bedrooms, 1 bathroom
            [3000, 4, 3]   # 3000 sqft, 4 bedrooms, 3 bathrooms
        ])
        
        # Scale sample features using the same scaler
        sample_features_scaled = scaler.transform(sample_features)
        sample_predictions = model.predict(sample_features_scaled)
        
        print("\nSample Predictions:")
        for i, (features, prediction) in enumerate(zip(sample_features, sample_predictions)):
            print(f"Sample {i+1}: {features[0]:.0f} sqft, {features[1]:.0f} bedrooms, "
                  f"{features[2]:.0f} bathrooms -> Predicted Price: ${prediction:,.2f}")
        
        print("\n" + "="*60)
        print("Model training and evaluation completed successfully!")
        print("="*60)
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
