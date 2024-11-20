import os
import pandas as pd
import joblib
from preprocess import preprocess_data
from feature_selection import feature_selection_pipeline
from base_models import train_models
from meta_learner import train_meta_learner_with_tuning, load_base_model_predictions
from evaluation import evaluate_model, print_evaluation_results
from sklearn.model_selection import train_test_split
from config import (
    BASE_MODELS_PATH, META_LEARNER_PATH, PREPROCESSED_DATA_PATH,FEATURE_SELECTION_PATH, PREPROCESSED_RENTALS_FILE,
    PREPROCESSED_TRANSACTIONS_FILE, RENTALS_TARGET, TRANSACTIONS_TARGET,
    META_LEARNER_RENTALS_MODEL, META_LEARNER_TRANSACTIONS_MODEL,
    RENTALS_META_LEARNER_HISTORY, TRANSACTIONS_META_LEARNER_HISTORY
)

def ensure_directories_exist():
    """
    Ensure that necessary directories exist for saving results.
    """
    os.makedirs(BASE_MODELS_PATH, exist_ok=True)
    os.makedirs(META_LEARNER_PATH, exist_ok=True)
    os.makedirs(PREPROCESSED_DATA_PATH, exist_ok=True)
    os.makedirs(FEATURE_SELECTION_PATH, exist_ok=True)
    print(f"Directories ensured: {BASE_MODELS_PATH}, {META_LEARNER_PATH}, {PREPROCESSED_DATA_PATH}, {FEATURE_SELECTION_PATH}")


def save_intermediate_results(data, file_name):
    """
    Save intermediate results as CSV.
    """
    data.to_csv(file_name, index=False)
    print(f"Saved: {file_name}")

def save_model(model, file_name):
    """
    Save trained models.
    """
    joblib.dump(model, file_name)
    print(f"Model saved: {file_name}")

def evaluate_and_print(y_true, y_pred, model_name):
    """
    Evaluate a model and print its performance metrics.
    """
    results = evaluate_model(y_true, y_pred)
    print_evaluation_results(results, model_name=model_name)

def main():
    # Step 1: Ensure directories exist
    ensure_directories_exist()

    # Step 2: Load and preprocess data
    print("Loading and preprocessing data...")
    rentals_df, transactions_df = preprocess_data()

    save_intermediate_results(rentals_df, PREPROCESSED_RENTALS_FILE)
    save_intermediate_results(transactions_df, PREPROCESSED_TRANSACTIONS_FILE)

    # Step 3: Feature Selection
    print("Running feature selection...")
    rentals_features, rentals_summary = feature_selection_pipeline(rentals_df, RENTALS_TARGET)
    transactions_features, transactions_summary = feature_selection_pipeline(transactions_df, TRANSACTIONS_TARGET)

    # Step 4: Split Data for Training
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        rentals_df[rentals_features], rentals_df[RENTALS_TARGET], test_size=0.2, random_state=42
    )
    X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(
        transactions_df[transactions_features], transactions_df[TRANSACTIONS_TARGET], test_size=0.2, random_state=42
    )

    # Step 5: Train Base Models
    print("Training base models for rentals dataset...")
    base_models_r = train_models(rentals_df, RENTALS_TARGET, rentals_features, "rentals")
    print("Training base models for transactions dataset...")
    base_models_t = train_models(transactions_df, TRANSACTIONS_TARGET, transactions_features, "transactions")

    # Step 6: Evaluate Base Models
    print("Evaluating base models for rentals dataset...")
    rentals_feature_order = joblib.load(f"{BASE_MODELS_PATH}rentals_features.pkl")
    X_test_r = X_test_r[rentals_feature_order]
    for model_name, model in base_models_r.items():
        y_pred = model.predict(X_test_r)
        evaluate_and_print(y_test_r, y_pred, model_name=f"{model_name} (Rentals)")

    print("Evaluating base models for transactions dataset...")
    transactions_feature_order = joblib.load(f"{BASE_MODELS_PATH}transactions_features.pkl")
    X_test_t = X_test_t[transactions_feature_order]
    for model_name, model in base_models_t.items():
        y_pred = model.predict(X_test_t)
        evaluate_and_print(y_test_t, y_pred, model_name=f"{model_name} (Transactions)")

    # Step 7: Generate Predictions for Meta-Learner
    print("Generating predictions for meta-learner...")
    meta_X_train_r = load_base_model_predictions(base_models_r.values(), X_train_r)
    meta_X_test_r = load_base_model_predictions(base_models_r.values(), X_test_r)
    meta_X_train_t = load_base_model_predictions(base_models_t.values(), X_train_t)
    meta_X_test_t = load_base_model_predictions(base_models_t.values(), X_test_t)

    # Step 8: Train Meta-Learner
    print("Training meta-learner for rentals dataset...")
    meta_learner_r, history_r = train_meta_learner_with_tuning(meta_X_train_r, y_train_r, meta_X_test_r, y_test_r)
    meta_learner_r.save(META_LEARNER_RENTALS_MODEL)
    pd.DataFrame(history_r.history).to_csv(RENTALS_META_LEARNER_HISTORY, index=False)
    print("Meta-learner for rentals saved.")

    print("Training meta-learner for transactions dataset...")
    meta_learner_t, history_t = train_meta_learner_with_tuning(meta_X_train_t, y_train_t, meta_X_test_t, y_test_t)
    meta_learner_t.save(META_LEARNER_TRANSACTIONS_MODEL)
    pd.DataFrame(history_t.history).to_csv(TRANSACTIONS_META_LEARNER_HISTORY, index=False)
    print("Meta-learner for transactions saved.")

    # Step 9: Evaluate Meta-Learner
    print("Evaluating the meta-learner for rentals dataset...")
    meta_y_pred_r = meta_learner_r.predict(meta_X_test_r)
    evaluate_and_print(y_test_r, meta_y_pred_r, model_name="Meta-Learner (Rentals)")

    print("Evaluating the meta-learner for transactions dataset...")
    meta_y_pred_t = meta_learner_t.predict(meta_X_test_t)
    evaluate_and_print(y_test_t, meta_y_pred_t, model_name="Meta-Learner (Transactions)")

    print("Pipeline execution completed successfully.")

if __name__ == "__main__":
    main()
