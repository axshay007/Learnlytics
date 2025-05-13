"""
Student Performance Prediction Model Training
This script contains the code for training and evaluating multiple machine learning models
to predict student performance based on various features.
"""

# Data manipulation and visualization libraries
import numpy as np  # For numerical operations
import pandas as pd  # For data manipulation
import matplotlib.pyplot as plt  # For plotting
import seaborn as sns  # For statistical visualizations

# Machine Learning libraries
# Metrics for model evaluation
from sklearn.metrics import mean_squared_error, r2_score

# Various regression models to compare
from sklearn.neighbors import KNeighborsRegressor  # K-Nearest Neighbors
from sklearn.tree import DecisionTreeRegressor  # Decision Tree
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor  # Ensemble methods
from sklearn.svm import SVR  # Support Vector Regression
from sklearn.linear_model import LinearRegression, Ridge, Lasso  # Linear models

# Model evaluation metrics
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Hyperparameter tuning
from sklearn.model_selection import RandomizedSearchCV

# Advanced gradient boosting models
from catboost import CatBoostRegressor  # CatBoost for handling categorical features
from xgboost import XGBRegressor  # XGBoost for high performance

import warnings  # To handle warnings
warnings.filterwarnings('ignore')  # Suppress warnings for cleaner output

def load_and_prepare_data(file_path):
    """
    Load and prepare the dataset for modeling.
    
    Args:
        file_path (str): Path to the CSV file containing student data
        
    Returns:
        tuple: (X, y) where X is the feature matrix and y is the target variable
    """
    # Load the dataset
    df = pd.read_csv(file_path)
    
    # Separate features (X) and target variable (y)
    # We're predicting math_score, so it's our target variable
    X = df.drop(columns=['math_score'], axis=1)  # Features
    y = df['math_score']  # Target variable
    
    return X, y

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """
    Train and evaluate multiple machine learning models.
    
    Args:
        X_train, X_test: Training and test feature matrices
        y_train, y_test: Training and test target variables
        
    Returns:
        dict: Dictionary containing model performance metrics
    """
    # Dictionary to store model performance metrics
    models = {
        'Linear Regression': LinearRegression(),  # Simple linear model
        'Ridge Regression': Ridge(),  # Linear model with L2 regularization
        'Lasso Regression': Lasso(),  # Linear model with L1 regularization
        'Decision Tree': DecisionTreeRegressor(),  # Non-linear tree-based model
        'Random Forest': RandomForestRegressor(),  # Ensemble of decision trees
        'XGBoost': XGBRegressor(),  # Gradient boosting with advanced features
        'CatBoost': CatBoostRegressor(verbose=False)  # Gradient boosting with categorical feature handling
    }
    
    # Dictionary to store model scores
    model_scores = {}
    
    # Train and evaluate each model
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        r2 = r2_score(y_test, y_pred)  # R-squared score
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))  # Root Mean Square Error
        mae = mean_absolute_error(y_test, y_pred)  # Mean Absolute Error
        
        # Store scores
        model_scores[name] = {
            'R2 Score': r2,
            'RMSE': rmse,
            'MAE': mae
        }
        
        print(f"{name} Performance:")
        print(f"R2 Score: {r2:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
    
    return model_scores

def visualize_model_comparison(model_scores):
    """
    Create visualizations to compare model performance.
    
    Args:
        model_scores (dict): Dictionary containing model performance metrics
    """
    plt.figure(figsize=(15, 6))
    
    # R2 Score comparison
    plt.subplot(1, 3, 1)
    r2_scores = [scores['R2 Score'] for scores in model_scores.values()]
    plt.bar(model_scores.keys(), r2_scores)
    plt.title('R2 Score Comparison')
    plt.xticks(rotation=45)
    
    # RMSE comparison
    plt.subplot(1, 3, 2)
    rmse_scores = [scores['RMSE'] for scores in model_scores.values()]
    plt.bar(model_scores.keys(), rmse_scores)
    plt.title('RMSE Comparison')
    plt.xticks(rotation=45)
    
    # MAE comparison
    plt.subplot(1, 3, 3)
    mae_scores = [scores['MAE'] for scores in model_scores.values()]
    plt.bar(model_scores.keys(), mae_scores)
    plt.title('MAE Comparison')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()

def tune_final_model(X_train, y_train):
    """
    Perform hyperparameter tuning for the final model (CatBoost).
    
    Args:
        X_train, y_train: Training data
        
    Returns:
        CatBoostRegressor: Tuned model with best parameters
    """
    # Select CatBoost as the final model
    final_model = CatBoostRegressor(verbose=False)
    
    # Define hyperparameter grid
    param_grid = {
        'iterations': [100, 200, 300],  # Number of trees
        'learning_rate': [0.01, 0.05, 0.1],  # Step size for gradient descent
        'depth': [4, 6, 8],  # Maximum depth of trees
        'l2_leaf_reg': [1, 3, 5]  # L2 regularization
    }
    
    # Perform hyperparameter tuning
    grid_search = RandomizedSearchCV(
        final_model,
        param_grid,
        cv=5,  # 5-fold cross-validation
        n_iter=10,  # Number of parameter settings to try
        scoring='r2',  # Use R2 score for evaluation
        random_state=42  # For reproducibility
    )
    
    # Fit the model with best parameters
    grid_search.fit(X_train, y_train)
    
    # Print best parameters and score
    print("Best Parameters:", grid_search.best_params_)
    print("Best R2 Score:", grid_search.best_score_)
    
    return grid_search.best_estimator_

def main():
    """
    Main function to run the model training and evaluation pipeline.
    """
    # Load and prepare data
    X, y = load_and_prepare_data('data/stud.csv')
    
    # Split data into training and test sets
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train and evaluate models
    model_scores = train_and_evaluate_models(X_train, X_test, y_train, y_test)
    
    # Visualize model comparison
    visualize_model_comparison(model_scores)
    
    # Tune final model
    best_model = tune_final_model(X_train, y_train)
    
    # Save the best model
    import joblib
    joblib.dump(best_model, 'artifacts/model.pkl')

if __name__ == "__main__":
    main() 