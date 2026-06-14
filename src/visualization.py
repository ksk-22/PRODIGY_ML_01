import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_actual_vs_predicted(y_test, y_pred):
    """
    Plot actual vs predicted values.
    
    Args:
        y_test: Actual target values
        y_pred: Predicted target values
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.6, color='blue', edgecolors='k')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
             'r--', lw=2, label='Perfect Prediction')
    plt.xlabel('Actual Price', fontsize=12)
    plt.ylabel('Predicted Price', fontsize=12)
    plt.title('Actual vs Predicted House Prices', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('actual_vs_predicted.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Plot saved as 'actual_vs_predicted.png'")

def plot_residuals(y_test, y_pred):
    """
    Plot residuals (errors) of predictions.
    
    Args:
        y_test: Actual target values
        y_pred: Predicted target values
    """
    residuals = y_test - y_pred
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Residual scatter plot
    axes[0].scatter(y_pred, residuals, alpha=0.6, color='green', edgecolors='k')
    axes[0].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[0].set_xlabel('Predicted Price', fontsize=12)
    axes[0].set_ylabel('Residuals', fontsize=12)
    axes[0].set_title('Residual Plot', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Residual distribution
    axes[1].hist(residuals, bins=20, color='purple', edgecolor='black', alpha=0.7)
    axes[1].set_xlabel('Residuals', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('Distribution of Residuals', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('residuals_plot.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Plot saved as 'residuals_plot.png'")

def plot_feature_importance(feature_names, coefficients):
    """
    Plot feature importance (coefficients).
    
    Args:
        feature_names (list): Names of features
        coefficients (np.array): Model coefficients
    """
    plt.figure(figsize=(10, 6))
    colors = ['red' if x < 0 else 'green' for x in coefficients]
    plt.barh(feature_names, coefficients, color=colors, alpha=0.7, edgecolor='black')
    plt.xlabel('Coefficient Value', fontsize=12)
    plt.title('Feature Importance (Model Coefficients)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Plot saved as 'feature_importance.png'")

def plot_data_distribution(data, feature_columns, target_column):
    """
    Plot distribution of features and target variable.
    
    Args:
        data (pd.DataFrame): Input data
        feature_columns (list): Names of feature columns
        target_column (str): Name of target column
    """
    n_features = len(feature_columns) + 1
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    # Plot feature distributions
    for idx, feature in enumerate(feature_columns):
        axes[idx].hist(data[feature], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        axes[idx].set_xlabel(feature, fontsize=11)
        axes[idx].set_ylabel('Frequency', fontsize=11)
        axes[idx].set_title(f'Distribution of {feature}', fontsize=12, fontweight='bold')
        axes[idx].grid(True, alpha=0.3, axis='y')
    
    # Plot target distribution
    axes[3].hist(data[target_column], bins=30, color='lightcoral', edgecolor='black', alpha=0.7)
    axes[3].set_xlabel(target_column, fontsize=11)
    axes[3].set_ylabel('Frequency', fontsize=11)
    axes[3].set_title(f'Distribution of {target_column}', fontsize=12, fontweight='bold')
    axes[3].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('data_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Plot saved as 'data_distribution.png'")
