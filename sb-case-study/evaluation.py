import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate_model(y_true, y_pred):
    """
    Evaluate a regression model using RMSE, MAE, and R² metrics.
    
    Parameters:
    - y_true (array-like): True target values.
    - y_pred (array-like): Predicted target values.

    Returns:
    - dict: A dictionary containing RMSE, MAE, and R² scores.
    """
    # Calculate evaluation metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # Return results as a dictionary
    return {
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2
    }

def print_evaluation_results(results, model_name="Model"):
    """
    Print evaluation results in a readable format.
    
    Parameters:
    - results (dict): Dictionary containing evaluation metrics.
    - model_name (str): Name of the model being evaluated.
    """
    print(f"Evaluation Metrics for {model_name}:")
    print(f"  RMSE: {results['RMSE']:.4f}")
    print(f"  MAE:  {results['MAE']:.4f}")
    print(f"  R²:   {results['R2']:.4f}")

# Example Usage
if __name__ == "__main__":
    # Example ground truth and predictions
    y_true = [3.5, 2.8, 4.0, 5.1, 6.2]
    y_pred = [3.4, 2.9, 4.1, 5.0, 6.3]

    # Evaluate the model
    results = evaluate_model(y_true, y_pred)
    
    # Print the results
    print_evaluation_results(results, model_name="Example Model")
