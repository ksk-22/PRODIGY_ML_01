from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np

class HousePricePredictor:
    """
    Linear Regression model for predicting house prices.
    """
    
    def __init__(self):
        """
        Initialize the linear regression model.
        """
        self.model = LinearRegression()
        self.is_trained = False
        self.coefficients = None
        self.intercept = None
        
    def train(self, X_train, y_train):
        """
        Train the linear regression model.
        
        Args:
            X_train: Training features
            y_train: Training target values
        """
        self.model.fit(X_train, y_train)
        self.coefficients = self.model.coef_
        self.intercept = self.model.intercept_
        self.is_trained = True
        print("Model trained successfully.")
        
    def predict(self, X):
        """
        Make predictions using the trained model.
        
        Args:
            X: Features for prediction
            
        Returns:
            np.array: Predicted values
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions.")
        return self.model.predict(X)
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on test data.
        
        Args:
            X_test: Testing features
            y_test: Testing target values
            
        Returns:
            dict: Dictionary containing performance metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation.")
        
        y_pred = self.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        metrics = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2_Score': r2
        }
        
        return metrics, y_pred
    
    def display_coefficients(self, feature_names):
        """
        Display model coefficients with feature names.
        
        Args:
            feature_names (list): Names of features
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first.")
        
        print("\n=== Model Coefficients ===")
        for name, coef in zip(feature_names, self.coefficients):
            print(f"{name}: {coef:.4f}")
        print(f"Intercept: {self.intercept:.4f}")
    
    def display_metrics(self, metrics):
        """
        Display evaluation metrics.
        
        Args:
            metrics (dict): Dictionary of performance metrics
        """
        print("\n=== Model Performance Metrics ===")
        for metric_name, metric_value in metrics.items():
            print(f"{metric_name}: {metric_value:.4f}")
