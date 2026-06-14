# Linear Regression Model for House Price Prediction

## Project Overview
This project implements a linear regression model to predict house prices based on three features:
- **Square Footage** (size of the house)
- **Number of Bedrooms**
- **Number of Bathrooms**

## Objective
Build and train a linear regression model that can accurately predict house prices using the above features.

## Tech Stack
- Python 3.x
- pandas - Data manipulation and analysis
- scikit-learn - Machine learning library
- NumPy - Numerical computations
- matplotlib - Data visualization
- seaborn - Statistical data visualization

## Project Structure
```
├── README.md
├── requirements.txt
├── data/
│   └── housing_data.csv
├── notebooks/
│   └── analysis.ipynb
└── src/
    ├── __init__.py
    ├── model.py
    ├── data_preprocessing.py
    └── visualization.py
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/ksk-22/PRODIGY_ML_01.git
cd PRODIGY_ML_01
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the main script to train and evaluate the model:
```bash
python main.py
```

## Model Performance
- Mean Squared Error (MSE)
- R² Score
- Root Mean Squared Error (RMSE)

## Results
The trained model will output:
- Model coefficients for each feature
- Model intercept
- Performance metrics on test data
- Predictions for sample data

## Author
Korrapati Sai Kamal
