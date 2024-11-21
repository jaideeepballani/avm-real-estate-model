  # Automated Valuation Model (AVM)

This repository contains the implementation of an Automated Valuation Model (AVM) for real estate pricing. The AVM leverages machine learning techniques, including Random Forest, XGBoost, and meta-learners, to estimate property price and rental values based on various features such as property size, contract length, location, and transaction history.

---

## Instructions
1. Clone the Repository:
   - Clone this repository to your local system using:
     ```bash
     git clone https://github.com/jaideeepballani/avm-real-estate-model.git
     cd <repository-directory>
     ```

2. Set Up the Environment:
   - Ensure all dependencies are installed as per the instructions in the [Environment Setup](#environment-setup) section.

3. Prepare the Data:
   - Follow the [Data Requirements](#data-requirements) section to ensure the input datasets are formatted and cleaned appropriately.

4. Train the Model:
   - Use the steps provided in the [Training the Model](#training-the-model) section to train and evaluate the model.

5. Make Predictions:
   - Refer to the [Making Predictions](#making-predictions-using-the-trained-model) section to generate predictions with the trained model.

---

## Environment Setup

### Requirements
- Python 3.11 or later
- Required libraries:
  - pandas
  - sweetviz(EDA)
  - os
  - numpy
  - scikit-learn
  - xgboost
  - matplotlib
  - plotly
  - alibi (for ALE plots)
  - joblib
  - Any additional libraries as specified in `requirements.txt`

### Steps to Set Up
1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Verify the setup:
   ```bash
   python --version
   pip list
   ```

---

## Data Requirements

The AVM model currently runs on two datasets:
1. Rentals Dataset:
   - Contains rental property information.
   - Target Variable: `annual_amount`.

2. Transactions Dataset:
   - Includes property sales transaction details.
   - Target Variable: `amount`.

This can be changed using `config.py` file which contains the paths, configurations, model parameter grids etc.



### Data Cleaning and Preprocessing
- Use the `preprocess.py` script to handle NaN values, remove outliers, and engineer features.
- Ensure categorical variables are one-hot encoded before training the model.

### Feature Selection
- The script feature selection currently runs correlation analysis 
- Other suggested methods like Recursive Feature Elimination, Univariate Feature Selection
- Feature Importance was used in separate Jupyter Notebook for base model training. You can navigate to the Jupyter Notebook through the `notebooks` folder in the repo.

### Expected Columns
- `rentals_df`: Columns such as `property_size_sqm`, `location`, `contract_start_date`, `contract_end_date`, etc.
- `transactions_df`: Columns like `property_type`, `amount`, etc.

---

## Training the Model

1. Prepare the Data:
   - Split the datasets into training and testing sets using the `prepare_data` function in `base_models.py`.

2. Run Training:
   - Use the following commands to train models:
     ```bash
     python base_models.py
     ```
   - This script trains models (Random Forest and XGBoost) for both datasets using GridSearchCV and saves the best models. Currently, SVR is not being used but the code is available and commented out.

3. Meta Learner
   - This AVM utilises a Meta Learner to combine the predictions of our base models(Random Forest, XGBoost, SVR etc)
   to improve the performance.

3. Evaluate Models:
   - Evaluation metrics include RMSE, MAE, R². These are being calculated in `evaluation.py`
   - Feature importance plots and ALE plots are generated for interpretability in separate Jupyter notebooks.

4. Saved Models:
   - Trained models are saved in the `results/` directory with appropriate naming conventions.

---

## Making Predictions Using the Trained Model

1. Load the Trained Model:
   - Use `joblib` to load the pre-trained model:
     ```python
     from joblib import load
     model = load('results/base_models/rentals_random_forest.pkl')
     ```

2. Prepare Input Data:
   - Ensure the input data follows the same preprocessing steps as the training data.
   - Use the `preprocess.py` script for consistency.

3. Generate Predictions:
   - Example:
     ```python
     predictions = model.predict(X_test)
     print(predictions)
     ```

4. Save Predictions:
   - Use pandas to save predictions to a CSV file:
     ```python
     import pandas as pd
     pd.DataFrame({'Predicted Values': predictions}).to_csv('predictions.csv', index=False)
     ```

---

## Running the entire pipeline

- The entrypoint of the AVM is `main.py`
  ```bash
  python main.py 
  ```
- Ensure you have installed the dependencies in the `requirements.txt` file and the Instructions section of this README. 
- Ensure you have tweaked everything you need to in `config.py`
- Dockerfile `dockerfile.yaml` is also provided to containerise this application

---

## Additional Notes

### Testing
- Unit tests are defined in `test.py`. They can be used to stress test each functionality of the pipeline

### Containerization
- Docker image file is provided - `dockerfile.yaml`. This can be used to containerize the application.

### Features
- For rentals, a derived feature `contract_length` is computed as:
  ```python
  df['contract_length'] = ((df.contract_end_date - df.contract_start_date) / np.timedelta64(1, 'D')) / 30
  ```

### Debugging
- Logging and debugging messages are logged wherever relevant.

### Future Work
- Incorporate ensemble meta-learners and advanced feature engineering for improved accuracy.
- Serve predictions using an API
- Web interface to interact with AVM and generate predictions

---
