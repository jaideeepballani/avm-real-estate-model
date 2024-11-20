import unittest
import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from config import (
    BASE_MODELS_PATH, 
    RENTALS_TARGET, 
    TRANSACTIONS_TARGET, 
    RESULTS_PATH
)
from preprocess import preprocess_data
from feature_selection import feature_selection_pipeline
from base_models import train_models
from evaluation import evaluate_model


class TestPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Set up shared resources for the tests.
        """
        print("Setting up test environment...")
        cls.rentals_data, cls.transactions_data, _, _, _, _ = preprocess_data()
        cls.rentals_target = RENTALS_TARGET
        cls.transactions_target = TRANSACTIONS_TARGET

        # Create results directory if it doesn't exist
        os.makedirs(RESULTS_PATH, exist_ok=True)

    def test_preprocessing(self):
        """
        Test that preprocessing outputs valid dataframes.
        """
        self.assertIsInstance(self.rentals_data, pd.DataFrame, "Preprocessed rentals data should be a DataFrame")
        self.assertIsInstance(self.transactions_data, pd.DataFrame, "Preprocessed transactions data should be a DataFrame")
        self.assertFalse(self.rentals_data.empty, "Preprocessed rentals data should not be empty")
        self.assertFalse(self.transactions_data.empty, "Preprocessed transactions data should not be empty")

    def test_feature_selection(self):
        """
        Test feature selection pipeline for rentals data.
        """
        features, summary = feature_selection_pipeline(self.rentals_data, self.rentals_target)
        self.assertIsInstance(features, list, "Feature selection should return a list of selected features")
        self.assertTrue(len(features) > 0, "Feature selection should return at least one feature")
        self.assertIsInstance(summary, pd.DataFrame, "Feature selection summary should be a DataFrame")
        self.assertFalse(summary.empty, "Feature selection summary should not be empty")

    def test_model_training(self):
        """
        Test training for Random Forest on rentals dataset.
        """
        features, _ = feature_selection_pipeline(self.rentals_data, self.rentals_target)
        models = train_models(self.rentals_data, self.rentals_target, features, dataset_name="rentals_test")
        
        self.assertIn("Random Forest", models, "Trained models should include Random Forest")
        self.assertIsInstance(models["Random Forest"], RandomForestRegressor, "Random Forest model should be an instance of RandomForestRegressor")

    def test_model_saving(self):
        """
        Test that models are saved correctly.
        """
        rf_model_path = f"{BASE_MODELS_PATH}rentals_test_random_forest.pkl"
        self.assertTrue(os.path.exists(rf_model_path), "Random Forest model file should exist after training")

    def test_model_evaluation(self):
        """
        Test evaluation metrics.
        """
        dummy_y_true = [3, -0.5, 2, 7]
        dummy_y_pred = [2.5, 0.0, 2, 8]
        results = evaluate_model(dummy_y_true, dummy_y_pred)
        self.assertIn("RMSE", results, "Evaluation results should include RMSE")
        self.assertIn("R2", results, "Evaluation results should include R2")
        self.assertIn("MAE", results, "Evaluation results should include MAE")
        self.assertGreaterEqual(results["R2"], 0, "R2 score should be at least 0")

    @classmethod
    def tearDownClass(cls):
        """
        Clean up any test files created during testing.
        """
        print("Cleaning up test environment...")
        for file in os.listdir(BASE_MODELS_PATH):
            if "rentals_test" in file:
                os.remove(os.path.join(BASE_MODELS_PATH, file))


if __name__ == "__main__":
    unittest.main()
