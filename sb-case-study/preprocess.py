# preprocess.py

import pandas as pd
#from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np

from config import RENTALS_PATH, TRANSACTIONS_PATH

def load_data():
    # Load both CSV files
    rentals_df = pd.read_csv(RENTALS_PATH)
    transactions_df = pd.read_csv(TRANSACTIONS_PATH)
    return rentals_df, transactions_df

def inspect_data(rentals_df, transactions_df):
    print("=== Rentals DataFrame ===")
    print(rentals_df.head())
    print("\nRentals DataFrame Info:")
    print(rentals_df.info())
    print("\nRentals DataFrame Description:")
    print(rentals_df.describe())
    
    print("\n" + "="*50 + "\n")
    
    print("=== Transactions DataFrame ===")
    print(transactions_df.head())
    print("\nTransactions DataFrame Info:")
    print(transactions_df.info())
    print("\nTransactions DataFrame Description:")
    print(transactions_df.describe())

def handle_nans_outliers(df, dataset_name=None):
    # Drop low-quality columns with very few non-null values
    # derive contract length feature
    low_quality_cols = ['building_age','ejari_contract_number','registration_date','version_number','version_text','contract_amount','is_freehold','parcel_id','property_id','land_property_id','property_type_ar','property_subtype_ar','property_usage_ar','property_usage_id','project_name_ar','area_ar','area_id','nearest_landmark_ar','nearest_metro_ar','nearest_mall_ar','master_project_en','parking','project_name_en','master_project_ar','req_from','req_to','entry_id','meta_ts','ejari_property_type_id','ejari_property_sub_type_id','transaction_datetime', 'transaction_subtype_id','transaction_number','transaction_type_id','property_usage_id','parcel_id','transaction_size_sqm','property_id','property_type_ar','property_type_id','property_subtype_ar','property_subtype_id','rooms_ar','project_name_ar','area_ar','area_id','nearest_landmark_ar','nearest_metro_ar','nearest_mall_ar','master_project_ar','req_from','req_to','entry_id','meta_ts']
    df = df.drop(columns=[col for col in low_quality_cols if col in df.columns], errors='ignore')
    
        # Check if dataset_name is provided
    if not dataset_name:
        raise ValueError("The 'dataset_name' parameter must be specified as 'rentals_df' or 'transactions_df'.")

    # Handle rentals_df
    if dataset_name == 'rentals_df':
        # Count rows with NaNs in the target
        nans_in_target = df['annual_amount'].isna().sum()
        print(f"Number of NaNs in 'annual_amount' (rentals_df): {nans_in_target}")
        
        # Remove rows with NaNs in the target
        df = df.dropna(subset=['annual_amount'])
        
        # Remove rows with annual_amount > 2,500,000
        outliers_removed = (df['annual_amount'] > 2500000).sum()
        print(f"Number of rows removed with 'annual_amount' > 2,500,000: {outliers_removed}")
        df = df[df['annual_amount'] <= 2500000]

    # Handle transactions_df
    elif dataset_name == 'transactions_df':
        # Count rows with NaNs in the target
        nans_in_target = df['amount'].isna().sum()
        print(f"Number of NaNs in 'amount' (transactions_df): {nans_in_target}")
        
        # Remove rows with NaNs in the target
        df = df.dropna(subset=['amount'])
        
        # Remove rows with amount < 50,000 or amount > 50,000,000
        outliers_removed = ((df['amount'] < 50000) | (df['amount'] > 50000000)).sum()
        print(f"Number of rows removed with 'amount' < 50,000 or > 50,000,000: {outliers_removed}")
        df = df[(df['amount'] >= 50000) & (df['amount'] <= 50000000)]

    else:
        raise ValueError("Invalid dataset_name. Use 'rentals_df' or 'transactions_df'.")
    
    return df

def feature_engineering(df, date_cols=None, dataset_name=None):
    # Convert date columns to datetime and create new date-based features
    # derive contract length feature
    if date_cols:
        for col in date_cols:
            if col in df.columns:

                df[col] = pd.to_datetime(df[col], errors='coerce') 
                df[f'{col}_year'] = df[col].dt.year
                df[f'{col}_month'] = df[col].dt.month
                df[f'{col}_day'] = df[col].dt.day
            
    if dataset_name=='rentals_df':
        df['contract_length'] = ((df.contract_end_date - df.contract_start_date) / np.timedelta64(1, 'D')) / 30
        df['contract_length'] = df['contract_length'].astype(int)

    if date_cols:
        df = df.drop(columns=[col for col in date_cols if col in df.columns], errors='ignore')
    
    return df



def one_hot_encode(df, categorical_columns=None, drop_first=True):

    if categorical_columns is None:
        # Automatically detect categorical columns
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Ensure the columns exist in the DataFrame
    missing_columns = [col for col in categorical_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Columns not found in DataFrame: {missing_columns}")
    
    # Perform one-hot encoding
    df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=drop_first)
    
    return df_encoded

'''def encode_categorical(df):
    # Encode categorical variables with LabelEncoder
    label_encoders = {}
    for column in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le
    return df, label_encoders

def scale_numerical_features(df):
    # Scale numerical features to a standard scale
    scaler = StandardScaler()
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
    return df, scaler''' 

def preprocess_data():
    # Load data
    rentals_df, transactions_df = load_data()
    
    # Inspect data
    inspect_data(rentals_df, transactions_df)
    
    # Define date columns
    rentals_date_cols = ['contract_start_date', 'contract_end_date']
    #transactions_date_cols = ['transaction_datetime']
    
    # Process rentals data
    rentals_df = handle_nans_outliers(rentals_df, dataset_name='rentals_df')
    rentals_df = feature_engineering(rentals_df, rentals_date_cols,dataset_name='rentals_df')
    rentals_df = one_hot_encode(rentals_df)
    #rentals_df, rentals_scaler = scale_numerical_features(rentals_df)
    
    # Process transactions data
    transactions_df = handle_nans_outliers(transactions_df,dataset_name='transactions_df')
    transactions_df = feature_engineering(transactions_df, None, dataset_name='transactions_df')
    transactions_df = one_hot_encode(transactions_df)
    #transactions_df = scale_numerical_features(transactions_df)
    
    return (rentals_df, transactions_df)

# Running inspection and preprocessing
if __name__ == "__main__":
    rentals_df, transactions_df = preprocess_data()
    print("\nPreprocessing complete.")
