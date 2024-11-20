# base_models.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from skopt import BayesSearchCV
from sklearn.model_selection import GridSearchCV
from skopt.space import Real, Integer
import joblib
import os
from config import BASE_MODELS_PATH, RENTALS_RANDOM_FOREST_PARAMS,TRANSACTIONS_RANDOM_FOREST_PARAMS, RENTALS_XGBOOST_PARAMS, TRANSACTIONS_XGBOOST_PARAMS, SVR_PARAMS


def prepare_data(df, target_col, selected_features):
    """
    Prepares the data for training by selecting features and splitting into training and testing sets.

    Args:
        df (pd.DataFrame): Full dataframe containing features and the target variable.
        target_col (str): The target column name.
        selected_features (list): List of selected features to use.

    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    print("Debugging prepare_data function:")
    print(f"Input dataframe columns: {df.columns.tolist()}")
    print(f"Target column: {target_col}")
    print(f"Selected features: {selected_features}")

    # Ensure selected features exist in df
    missing_features = [feature for feature in selected_features if feature not in df.columns]
    if missing_features:
        raise KeyError(f"The following selected features are missing in the dataframe: {missing_features}")

    # Extract features (X) and target (y)
    X = df[selected_features]
    y = df[target_col]

    print(f"Feature matrix shape: {X.shape}")
    print(f"Target variable shape: {y.shape}")

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test



def random_forest_model(X_train, y_train, dataset_name):
    """
    Train Random Forest model with Grid Search Cross Validation (Optimized for performance).
    """
    print("Training Random Forest with Grid Search Cross Validation...")
    rf = RandomForestRegressor(random_state=42)
    # Use RENTAL_PARAMS or TRANSACTION_PARAMS based on dataset_name
    if dataset_name == 'rentals_df':
        param_grid = RENTALS_RANDOM_FOREST_PARAMS
    elif dataset_name == 'transactions_df':
        param_grid = TRANSACTIONS_RANDOM_FOREST_PARAMS
    else:
        raise ValueError("Invalid dataset_name. Use 'rentals_df' or 'transactions_df'.")
    grid_search = GridSearchCV(rf, param_grid, scoring="neg_mean_squared_error", cv=5, n_jobs=-1, verbose=1) #Opted for GridSearchCV over BayesSearchCV
    grid_search.fit(X_train, y_train)
    print(f"Best parameters for Random Forest: {grid_search.best_params_}")
    return grid_search.best_estimator_

def xgboost_model(X_train, y_train,dataset_name):
    """
    Train XGBoost model with Grid Search Cross Validation.
    """
    print("Training XGBoost with Grid Search Cross Validation...")
    xgb = XGBRegressor(random_state=42, objective='reg:squarederror')
    # Use RENTAL_PARAMS or TRANSACTION_PARAMS based on dataset_name
    if dataset_name == 'rentals_df':
        param_grid = RENTALS_XGBOOST_PARAMS
    elif dataset_name == 'transactions_df':
        param_grid = TRANSACTIONS_XGBOOST_PARAMS
    else:
        raise ValueError("Invalid dataset_name. Use 'rentals_df' or 'transactions_df'.")
    grid_search = GridSearchCV(xgb, param_grid, scoring="neg_mean_squared_error", cv=5, n_jobs=-1, verbose=1) #Opted for GridSearchCV over BayesSearchCV
    grid_search.fit(X_train, y_train)
    print(f"Best parameters for XGBoost: {grid_search.best_params_}")
    return grid_search.best_estimator_

'''def svr_model(X_train, y_train):
    """
    Train SVR model with Bayesian Optimization (Optimized for performance).
    """
    print("Training Support Vector Regression (SVR) with Bayesian Optimization...")
    svr = SVR()
    param_grid = {
        'C': Real(0.1, 1.0, prior='log-uniform'),  # Narrow range
        'epsilon': Real(0.01, 0.1, prior='log-uniform'),  # Narrow range
        'kernel': ['linear']  # Use a simpler kernel
    }
    bayes_search = BayesSearchCV(svr, param_grid, n_iter=10, cv=3, random_state=42, n_jobs=-1, verbose=1)
    
    # Limit dataset size for SVR
    X_train, _, y_train, _ = train_test_split(X_train, y_train, test_size=0.8, random_state=42) # UNDERSTAND THIS 
    
    bayes_search.fit(X_train, y_train)
    print(f"Best parameters for SVR: {bayes_search.best_params_}")
    return bayes_search.best_estimator_'''

def train_models(df, target_col, selected_features, dataset_name):
    """
    Train models for a given dataset, or load saved models if they exist.
    Returns a dictionary of models.
    """
    # Prepare data
    X_train, X_test, y_train, y_test = prepare_data(df, target_col, selected_features)

    # Initialize dictionary to hold models
    models = {}

    # Define file paths using config.py
    rf_model_path = f"{BASE_MODELS_PATH}{dataset_name}_random_forest.pkl"
    xgb_model_path = f"{BASE_MODELS_PATH}{dataset_name}_xgboost.pkl"
    svr_model_path = f"{BASE_MODELS_PATH}{dataset_name}_svr.pkl"
    feature_order_path = f"{BASE_MODELS_PATH}{dataset_name}_features.pkl"

    # Save the feature order used during training
    joblib.dump(selected_features, feature_order_path)
    print(f"Feature order saved for {dataset_name} at: {feature_order_path}")

    # Random Forest
    if os.path.exists(rf_model_path):
        print(f"Loading saved Random Forest model for {dataset_name}...")
        rf_model = joblib.load(rf_model_path)
    else:
        print(f"Training Random Forest model for {dataset_name}...")
        rf_model = random_forest_model(X_train, y_train,dataset_name)
        joblib.dump(rf_model, rf_model_path)
        print(f"Random Forest model saved at: {rf_model_path}")
    models["Random Forest"] = rf_model

    # XGBoost
    if os.path.exists(xgb_model_path):
        print(f"Loading saved XGBoost model for {dataset_name}...")
        xgb_model = joblib.load(xgb_model_path)
    else:
        print(f"Training XGBoost model for {dataset_name}...")
        xgb_model = xgboost_model(X_train, y_train,dataset_name)
        joblib.dump(xgb_model, xgb_model_path)
        print(f"XGBoost model saved at: {xgb_model_path}")
    models["XGBoost"] = xgb_model

    # Support Vector Regression (SVR)
    '''if os.path.exists(svr_model_path):
        print(f"Loading saved SVR model for {dataset_name}...")
        svr_model = joblib.load(svr_model_path)
    else:
        print(f"Training SVR model for {dataset_name}...")
        svr_model = svr_model(X_train, y_train)
        joblib.dump(svr_model, svr_model_path)
        print(f"SVR model saved at: {svr_model_path}")
    models["SVR"] = svr_model'''

    print(f"Models and feature order saved for {dataset_name} dataset.")
    return models



if __name__ == "__main__":
    from preprocess import preprocess_data
    from feature_selection import feature_selection_pipeline

    # Preprocess data
    rentals_df, transactions_df = preprocess_data()

    # Target columns and selected features
    rentals_target = 'annual_amount'
    transactions_target = 'amount'
    
    # Feature selection
    print("\nSelecting features for rentals...")
    rentals_features, _ = feature_selection_pipeline(rentals_df, rentals_target)
    
    print("\nSelecting features for transactions...")
    transactions_features, _ = feature_selection_pipeline(transactions_df, transactions_target)

    # Train models
    print("\nTraining models for rentals dataset...")
    train_models(rentals_df, rentals_target, rentals_features, "rentals_df")

    print("\nTraining models for transactions dataset...")
    train_models(transactions_df, transactions_target, transactions_features, "transactions_df")
