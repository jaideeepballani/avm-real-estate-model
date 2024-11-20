# meta_learner.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from keras_tuner import Hyperband
import joblib

def load_base_model_predictions(models, X):
    """
    Generate predictions from base models with aligned feature names.
    """
    predictions = []
    for model in models:
        # Align the test data features with the model's training features
        if hasattr(model, "feature_names_in_"):  # For sklearn models
            X_aligned = X[model.feature_names_in_]
        else:
            X_aligned = X
        predictions.append(model.predict(X_aligned))
    return np.column_stack(predictions)

def build_meta_learner(hp, input_dim):
    """
    Build a meta-learner model with hyperparameter tuning using keras-tuner.
    """
    model = Sequential()
    # Input layer
    model.add(Dense(
        units=hp.Int('units_input', min_value=32, max_value=128, step=16),
        activation='relu',
        input_dim=input_dim  # Dynamically set based on actual input features
    ))
    model.add(Dropout(hp.Float('dropout_input', min_value=0.1, max_value=0.5, step=0.1)))
    
    # Hidden layers
    for i in range(hp.Int('num_hidden_layers', 1, 3)):
        model.add(Dense(
            units=hp.Int(f'units_hidden_{i}', min_value=16, max_value=64, step=16),
            activation='relu'
        ))
        model.add(Dropout(hp.Float(f'dropout_hidden_{i}', min_value=0.1, max_value=0.5, step=0.1)))
    
    # Output layer
    model.add(Dense(1, activation='linear'))  # Single output for regression
    
    model.compile(
        optimizer=Adam(learning_rate=hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
        loss='mse',
        metrics=['mae']
    )
    return model


def train_meta_learner_with_tuning(meta_X_train, y_train, meta_X_test, y_test):
    """
    Train the meta-learner neural network with hyperparameter tuning.
    """
    print("Tuning the meta-learner...")
    
    input_dim = meta_X_train.shape[1]  # Number of features in the input data

    # Define the tuner
    tuner = Hyperband(
        lambda hp: build_meta_learner(hp, input_dim),
        objective='val_loss',
        max_epochs=50,
        factor=3,
        overwrite=True,
        directory='tuner_logs',
        project_name='meta_learner_tuning'
    )
    
    # Early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    # Run the tuner
    tuner.search(meta_X_train, y_train, validation_data=(meta_X_test, y_test), callbacks=[early_stopping])
    
    # Get the best hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print(f"Best hyperparameters: {best_hps.values}")
    
    # Build and train the best model
    best_model = tuner.hypermodel.build(best_hps)
    history = best_model.fit(
        meta_X_train, y_train,
        validation_data=(meta_X_test, y_test),
        epochs=100,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=1
    )
    return best_model, history

if __name__ == "__main__":
    from preprocess import preprocess_data
    from feature_selection import feature_selection_pipeline

    # Preprocess data
    rentals_df, transactions_df = preprocess_data()

    # Target columns and selected features
    rentals_target = 'contract_amount'
    transactions_target = 'amount'
    
    # Feature selection
    print("\nSelecting features for rentals...")
    rentals_features, _ = feature_selection_pipeline(rentals_df, rentals_target)
    
    print("\nSelecting features for transactions...")
    transactions_features, _ = feature_selection_pipeline(transactions_df, transactions_target)

    # Prepare training and test data for rentals
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        rentals_df[rentals_features], rentals_df[rentals_target], test_size=0.2, random_state=42)

    # Load base models for rentals
    rf_r = joblib.load("rentals_random_forest.pkl")
    xgb_r = joblib.load("rentals_xgboost.pkl")
    #svr_r = joblib.load("rentals_svr.pkl")
    base_models_r = [rf_r, xgb_r]

    # Generate predictions for rentals meta-learner
    meta_X_train_r = load_base_model_predictions(base_models_r, X_train_r)
    meta_X_test_r = load_base_model_predictions(base_models_r, X_test_r)

    # Train rentals meta-learner with hyperparameter optimization
    meta_learner_r, history_r = train_meta_learner_with_tuning(meta_X_train_r, y_train_r, meta_X_test_r, y_test_r)
    meta_learner_r.save("rentals_meta_learner_tuned.keras")
    print("Meta-learner saved for rentals dataset.")

    # Prepare training and test data for transactions
    X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(
        transactions_df[transactions_features], transactions_df[transactions_target], test_size=0.2, random_state=42)

    # Load base models for transactions
    rf_t = joblib.load("transactions_random_forest.pkl")
    xgb_t = joblib.load("transactions_xgboost.pkl")
    #svr_t = joblib.load("transactions_svr.pkl")
    base_models_t = [rf_t, xgb_t]

    # Generate predictions for transactions meta-learner
    meta_X_train_t = load_base_model_predictions(base_models_t, X_train_t)
    meta_X_test_t = load_base_model_predictions(base_models_t, X_test_t)

    # Train transactions meta-learner with hyperparameter optimization
    meta_learner_t, history_t = train_meta_learner_with_tuning(meta_X_train_t, y_train_t, meta_X_test_t, y_test_t)
    meta_learner_t.save("transactions_meta_learner_tuned.keras")
    print("Meta-learner saved for transactions dataset.")
