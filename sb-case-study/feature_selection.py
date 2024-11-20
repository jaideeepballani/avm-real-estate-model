# feature_selection.py

import pandas as pd
import numpy as np
#from sklearn.ensemble import RandomForestRegressor
#from sklearn.feature_selection import RFE, SelectKBest, f_regression
#from sklearn.model_selection import train_test_split

'''def drop_non_numeric_columns(df):
    """
    Drop non-numeric columns to ensure compatibility with models.
    """
    return df.select_dtypes(include=['int64', 'float64'])'''

def correlation_analysis(df, threshold=0.85):
    """
    Perform correlation analysis to drop highly correlated features.
    """
    print("Running correlation analysis...")
    corr_matrix = df.corr().abs()
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)]
    df = df.drop(columns=to_drop, errors='ignore')
    print(to_drop)
    print(f"Dropped {len(to_drop)} correlated features.")
    return df, to_drop

'''def feature_importance_tree(df, target_col): #use and explain ALE
    """
    Use a Random Forest model to identify feature importance.
    """
    print("Computing feature importance using Random Forest...")
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X = drop_non_numeric_columns(X)  # Ensure X contains only numeric data
    model = RandomForestRegressor(random_state=42, n_jobs=-1, max_depth=10)  # Limit depth for faster training
    model.fit(X, y)
    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    # PLOT IMPORTANCES - COVER OPTIONAL TASK 
    print("Feature importance computation complete.")
    return importances

def recursive_feature_elimination(df, target_col, n_features=10): # Explain why not used
    """
    Use Recursive Feature Elimination (RFE) to select top features.
    """
    print("Running Recursive Feature Elimination (RFE)...")
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X = drop_non_numeric_columns(X)  # Ensure X contains only numeric data
    model = RandomForestRegressor(random_state=42, n_jobs=-1, max_depth=5)  # Reduce complexity for speed
    rfe = RFE(model, n_features_to_select=n_features, step=10)  # Evaluate 10 features at a time
    rfe.fit(X, y)
    selected_features = [feature for feature, selected in zip(X.columns, rfe.support_) if selected]
    print(f"RFE selected {len(selected_features)} features.")
    return selected_features

def univariate_feature_selection(df, target_col, k=10):
    """
    Use univariate feature selection (e.g., ANOVA F-test) to select top k features.
    """
    print("Running univariate feature selection...")
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X = drop_non_numeric_columns(X)  # Ensure X contains only numeric data
    selector = SelectKBest(score_func=f_regression, k=k)
    selector.fit(X, y)
    selected_features = X.columns[selector.get_support()].tolist()
    print(f"Univariate selection chose {len(selected_features)} features.")
    return selected_features'''

def feature_selection_pipeline(df, target_col, sample_size=5000):
    """
    Combine all feature selection methods to determine final selected features. - NOT NEEDED
    """
    # Sample the dataset to improve performance
    print(f"Sampling {sample_size} rows from the dataset for feature selection...")
    df_sample = df.sample(n=min(sample_size, len(df)), random_state=42)
    
    # Drop non-numeric columns before feature selection
    #df_sample = drop_non_numeric_columns(df_sample)
    
    # Correlation Analysis
    df_corr, dropped_corr_features = correlation_analysis(df_sample)

    # Assuming df_corr is your DataFrame
    final_selected_features = df_corr.columns.tolist()
    
    # Feature Importance
    #importance_scores = feature_importance_tree(df_corr, target_col)
    #top_features_by_importance = importance_scores.head(10).index.tolist() #
    
    # Recursive Feature Elimination
    #top_features_by_rfe = recursive_feature_elimination(df_corr, target_col, n_features=10)
    
    # Univariate Feature Selection
    #top_features_by_univariate = univariate_feature_selection(df_corr, target_col, k=10)
    
    # Combine selected features
    #final_selected_features = list(set(top_features_by_importance + top_features_by_rfe + top_features_by_univariate))
    
    print("Feature selection pipeline complete.")
    return final_selected_features, '''{
        "Dropped Correlated Features": dropped_corr_features,
        "Top Features by Importance": top_features_by_importance,
        "Top Features by RFE": top_features_by_rfe,
        "Top Features by Univariate": top_features_by_univariate,
        "Final Selected Features": final_selected_features
    }'''

if __name__ == "__main__":
    # Example usage for rentals_df and transactions_df
    from preprocess import preprocess_data

    rentals_df, transactions_df, _, _, _, _ = preprocess_data()
    
    # Assume target columns for each dataset
    rentals_target = 'annual_amount'
    transactions_target = 'amount'
    
    # Feature selection for rentals
    print("\n=== Feature Selection for Rentals ===")
    rentals_features, rentals_summary = feature_selection_pipeline(rentals_df, rentals_target, sample_size=5000)
    print(rentals_summary)
    
    # Feature selection for transactions
    print("\n=== Feature Selection for Transactions ===")
    transactions_features, transactions_summary = feature_selection_pipeline(transactions_df, transactions_target, sample_size=5000)
    print(transactions_summary)
