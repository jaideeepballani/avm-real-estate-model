# Configuration for file paths
DATA_PATH = "data/"
RESULTS_PATH = "results/"
BASE_MODELS_PATH = RESULTS_PATH + "base_models/"
META_LEARNER_PATH = RESULTS_PATH + "meta_learner/"
PREPROCESSED_DATA_PATH = RESULTS_PATH + "preprocessed/"
FEATURE_SELECTION_PATH = RESULTS_PATH + "feature_selection/"

# File names and paths
RENTALS_FILE = "snp_dld_2024_rents.csv"
TRANSACTIONS_FILE = "snp_dld_2024_transactions.csv"
RENTALS_PATH = "data/snp_dld_2024_rents.csv"
TRANSACTIONS_PATH = "data/snp_dld_2024_transactions.csv"

# Preprocessed file names
PREPROCESSED_RENTALS_FILE = PREPROCESSED_DATA_PATH + "preprocessed_rentals.csv"
PREPROCESSED_TRANSACTIONS_FILE = PREPROCESSED_DATA_PATH + "preprocessed_transactions.csv"

# Feature selection outputs
FEATURE_SELECTION_SUMMARY_RENTALS = FEATURE_SELECTION_PATH + "rentals_features_summary.csv"
FEATURE_SELECTION_SUMMARY_TRANSACTIONS = FEATURE_SELECTION_PATH + "transactions_features_summary.csv"

# Meta-learner files
META_LEARNER_RENTALS_MODEL = META_LEARNER_PATH + "rentals_meta_learner.keras"
META_LEARNER_TRANSACTIONS_MODEL = META_LEARNER_PATH + "transactions_meta_learner.keras"

# Training history
RENTALS_META_LEARNER_HISTORY = RESULTS_PATH + "rentals_meta_learner_history.csv"
TRANSACTIONS_META_LEARNER_HISTORY = RESULTS_PATH + "transactions_meta_learner_history.csv"

# Model parameters for Base Models
RENTALS_RANDOM_FOREST_PARAMS = {
    "n_estimators": [500],
    "max_depth": [20],
    "min_samples_split": [5],
}

TRANSACTIONS_RANDOM_FOREST_PARAMS = {
    "n_estimators": [100],
    "max_depth": [50],
    "min_samples_split": [5],
}

RENTALS_XGBOOST_PARAMS = {
    "n_estimators": [500],
    "max_depth": [11],
    "learning_rate": [0.1],
}
TRANSACTIONS_XGBOOST_PARAMS = {
    "n_estimators": [200],
    "max_depth": [12],
    "learning_rate": [0.1],
}
SVR_PARAMS = {
    "C": [0.1, 1, 10],
    "epsilon": [0.01, 0.1, 1],
    "kernel": ["linear", "rbf"],
}

# Hyperparameter tuning for Meta-Learner
META_LEARNER_TUNING = {
    "units_input": [32, 64, 128],
    "dropout_input": [0.1, 0.3, 0.5],
    "num_hidden_layers": [1, 2, 3],
    "units_hidden": [16, 32, 64],
    "dropout_hidden": [0.1, 0.3, 0.5],
    "learning_rate": [0.0001, 0.001, 0.01],
    "epochs": 50,
    "batch_size": 64,
}

# Feature selection settings
CORRELATION_THRESHOLD = 0.85
NUM_TOP_FEATURES = 10

# Evaluation settings
METRICS = ["RMSE", "R2", "MAE"]

# Random seeds for reproducibility
SEED = 42

# Dataset-specific targets
RENTALS_TARGET = "annual_amount"
TRANSACTIONS_TARGET = "amount"
