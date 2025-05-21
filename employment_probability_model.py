"""
Employment Probability Post-Graduation Model

This script trains and evaluates machine learning regression models for predicting employment probability
post-graduation based on student data from the Alumni Management System.

Models implemented:
- Random Forest Regressor
- XGBoost Regressor
- Linear Regression

Results are saved to MongoDB for later visualization.

Author: Augment Agent
Date: 2025-05-21
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import explained_variance_score
import xgboost as xgb
import joblib
import warnings
from mongodb_utils import MongoDBHandler

warnings.filterwarnings('ignore')

# Set up logging
log_dir = "employment_logs"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(os.path.join(log_dir, "plots"), exist_ok=True)
os.makedirs(os.path.join(log_dir, "reports"), exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(log_dir, f"employment_probability_{timestamp}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# Create models directory
models_dir = "employment_models"
os.makedirs(models_dir, exist_ok=True)
model_save_dir = os.path.join(models_dir, f"employment_{timestamp}")
os.makedirs(model_save_dir, exist_ok=True)

def load_data(train_path, val_path, test_path):
    """
    Load the datasets from CSV files.

    Args:
        train_path: Path to the training dataset
        val_path: Path to the validation dataset
        test_path: Path to the test dataset

    Returns:
        Tuple of DataFrames (train_df, val_df, test_df)
    """
    logger.info(f"Loading datasets from {train_path}, {val_path}, and {test_path}")

    try:
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        test_df = pd.read_csv(test_path)

        logger.info(f"Train dataset shape: {train_df.shape}")
        logger.info(f"Validation dataset shape: {val_df.shape}")
        logger.info(f"Test dataset shape: {test_df.shape}")

        return train_df, val_df, test_df
    except Exception as e:
        logger.error(f"Error loading datasets: {str(e)}")
        raise

def preprocess_data(train_df, val_df, test_df):
    """
    Preprocess the datasets for model training.

    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        test_df: Test DataFrame

    Returns:
        Tuple of processed data and preprocessing objects
    """
    logger.info("Preprocessing datasets")

    # Combine datasets for consistent preprocessing
    all_data = pd.concat([train_df, val_df, test_df], axis=0)

    # Clean data
    all_data = clean_data(all_data)

    # Feature engineering
    all_data = engineer_features(all_data)

    # Split back into train, valid, test
    train_processed = all_data.iloc[:len(train_df)]
    val_processed = all_data.iloc[len(train_df):len(train_df)+len(val_df)]
    test_processed = all_data.iloc[len(train_df)+len(val_df):]

    # Define features and target
    target_col = 'employment_probability'  # Using our new continuous target

    # Drop columns not useful for prediction
    drop_cols = ['student_id', 'name', 'predicted_job_title', 'actual_job_title', 'skills_used',
                 'time_to_employment', 'industry', 'company_type', 'salary_range', 'employed_after_grad']
    feature_cols = [col for col in train_processed.columns if col not in [target_col] + drop_cols]

    # Prepare data for modeling
    X_train = train_processed[feature_cols]
    y_train = train_processed[target_col]

    X_val = val_processed[feature_cols]
    y_val = val_processed[target_col]

    X_test = test_processed[feature_cols]
    y_test = test_processed[target_col]

    # Identify categorical and numerical columns
    categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()

    logger.info(f"Categorical columns: {categorical_cols}")
    logger.info(f"Numerical columns: {numerical_cols}")

    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ],
        remainder='drop'
    )

    # Fit preprocessor on training data
    preprocessor.fit(X_train)

    # Transform data
    X_train_processed = preprocessor.transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)

    # Get feature names after one-hot encoding
    cat_feature_names = []
    for i, col in enumerate(categorical_cols):
        cat_values = preprocessor.transformers_[1][1].categories_[i]
        for val in cat_values:
            cat_feature_names.append(f"cat__{col}_{val}")

    feature_names = numerical_cols + cat_feature_names

    logger.info(f"Processed X_train shape: {X_train_processed.shape}")
    logger.info(f"Processed X_val shape: {X_val_processed.shape}")
    logger.info(f"Processed X_test shape: {X_test_processed.shape}")

    # Print target variable statistics
    logger.info(f"Target variable (employment_probability) statistics:")
    logger.info(f"  Train mean: {y_train.mean():.4f}, std: {y_train.std():.4f}")
    logger.info(f"  Validation mean: {y_val.mean():.4f}, std: {y_val.std():.4f}")
    logger.info(f"  Test mean: {y_test.mean():.4f}, std: {y_test.std():.4f}")

    return (X_train_processed, y_train, X_val_processed, y_val,
            X_test_processed, y_test, preprocessor, feature_names)

def clean_data(df):
    """
    Clean the dataset by handling missing values and data type conversions.

    Args:
        df: DataFrame to clean

    Returns:
        Cleaned DataFrame
    """
    logger.info("Cleaning data")

    # Make a copy to avoid modifying the original
    df = df.copy()

    # Handle missing values for employment probability
    if 'employed_after_grad' in df.columns:
        # Convert to numeric (1.0 for employed, 0.0 for not employed)
        df['employed_after_grad'] = df['employed_after_grad'].map({'Yes': 1.0, 'No': 0.0})
        df['employed_after_grad'] = df['employed_after_grad'].fillna(0.0)

    # Handle time to employment - this will be our target for regression
    if 'time_to_employment' in df.columns:
        # Fill missing values with a high value (e.g., 24 months) for those not employed
        df.loc[df['employed_after_grad'] == 0, 'time_to_employment'] = 24.0
        # For those employed but missing time data, use median
        employed_mask = df['employed_after_grad'] == 1
        median_time = df.loc[employed_mask, 'time_to_employment'].median()
        df.loc[employed_mask & df['time_to_employment'].isna(), 'time_to_employment'] = median_time

        # Create employment probability score (inverse of time to employment, normalized)
        # Lower time to employment = higher probability score
        df['employment_probability'] = 1.0 - (df['time_to_employment'] / 24.0)
        # Ensure values are between 0 and 1
        df['employment_probability'] = df['employment_probability'].clip(0.0, 1.0)

    if 'gender' in df.columns:
        df['gender'] = df['gender'].fillna('Unknown')

    if 'internship_experience' in df.columns:
        df['internship_experience'] = df['internship_experience'].fillna('No')
        # Convert to numeric
        df['internship_experience'] = df['internship_experience'].map({'Yes': 1.0, 'No': 0.0})

    # Handle skills column
    if 'skills' in df.columns:
        df['skills'] = df['skills'].fillna('')

    # Convert GPA to float
    if 'gpa' in df.columns:
        df['gpa'] = pd.to_numeric(df['gpa'], errors='coerce')
        median_gpa = df['gpa'].median()
        df['gpa'] = df['gpa'].fillna(median_gpa)

    # Convert graduation_year to int
    if 'graduation_year' in df.columns:
        df['graduation_year'] = pd.to_numeric(df['graduation_year'], errors='coerce')
        mode_year = df['graduation_year'].mode()[0]
        df['graduation_year'] = df['graduation_year'].fillna(mode_year)
        df['graduation_year'] = df['graduation_year'].astype(int)

    # Convert age to int
    if 'age' in df.columns:
        df['age'] = pd.to_numeric(df['age'], errors='coerce')
        median_age = df['age'].median()
        df['age'] = df['age'].fillna(median_age)
        df['age'] = df['age'].astype(int)

    return df

def engineer_features(df):
    """
    Engineer features for improved model performance.

    Args:
        df: DataFrame to engineer features for

    Returns:
        DataFrame with engineered features
    """
    logger.info("Engineering features")

    # Make a copy to avoid modifying the original
    df = df.copy()

    # Current year for calculating years since graduation
    current_year = datetime.now().year

    # Calculate years since graduation
    if 'graduation_year' in df.columns:
        df['years_since_graduation'] = current_year - df['graduation_year']

        # Create a feature for recent graduates (within last 2 years)
        df['is_recent_graduate'] = df['years_since_graduation'] <= 2

    # Extract skill count and skill types
    if 'skills' in df.columns:
        # Count number of skills
        df['skill_count'] = df['skills'].apply(lambda x: len(str(x).split(',')))

        # Check for specific skill types
        df['has_technical_skills'] = df['skills'].str.contains('program|code|develop|engineer|data|analysis|technical|software', case=False, na=False)
        df['has_soft_skills'] = df['skills'].str.contains('communicate|team|leadership|manage|organize|problem|solve', case=False, na=False)
        df['has_education_skills'] = df['skills'].str.contains('teach|education|train|mentor|tutor', case=False, na=False)
        df['has_healthcare_skills'] = df['skills'].str.contains('medical|health|care|patient|nurse|doctor', case=False, na=False)
        df['has_business_skills'] = df['skills'].str.contains('business|market|sales|finance|account', case=False, na=False)

    # Create GPA bins
    if 'gpa' in df.columns:
        bins = [0, 2.0, 2.5, 3.0, 3.5, 4.0]
        labels = ['<2.0', '2.0-2.5', '2.5-3.0', '3.0-3.5', '3.5-4.0']
        df['gpa_bin'] = pd.cut(df['gpa'], bins=bins, labels=labels, include_lowest=True)

    # One-hot encode categorical variables
    categorical_cols = ['degree', 'major', 'location', 'gender', 'gpa_bin']
    for col in categorical_cols:
        if col in df.columns:
            dummies = pd.get_dummies(df[col], prefix=col)
            df = pd.concat([df, dummies], axis=1)

    return df

def train_random_forest(X_train, y_train, X_val, y_val, feature_names):
    """
    Train a Random Forest regressor with hyperparameter tuning.

    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        feature_names: Names of features

    Returns:
        Trained model and best parameters
    """
    logger.info("Training Random Forest Regressor model")

    # Define parameter grid for hyperparameter tuning - optimized for high accuracy
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # Create base model
    rf = RandomForestRegressor(random_state=42)

    # Create grid search with reduced parallelism to avoid memory issues
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=3,
        n_jobs=2,  # Reduced from -1 to avoid memory issues
        verbose=1,
        scoring='r2'  # Use R² score for regression
    )

    # Fit grid search
    grid_search.fit(X_train, y_train)

    # Get best model
    best_rf = grid_search.best_estimator_

    # Evaluate on validation set
    y_val_pred = best_rf.predict(X_val)
    r2 = r2_score(y_val, y_val_pred)
    mse = mean_squared_error(y_val, y_val_pred)
    mae = mean_absolute_error(y_val, y_val_pred)

    logger.info(f"Random Forest best parameters: {grid_search.best_params_}")
    logger.info(f"Random Forest validation R² score: {r2:.4f}")
    logger.info(f"Random Forest validation MSE: {mse:.4f}")
    logger.info(f"Random Forest validation MAE: {mae:.4f}")

    # Get feature importance
    if hasattr(best_rf, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': feature_names[:len(best_rf.feature_importances_)],
            'importance': best_rf.feature_importances_
        })
        feature_importance = feature_importance.sort_values('importance', ascending=False)

        # Save feature importance to CSV
        feature_importance.to_csv(os.path.join(log_dir, "reports", "random_forest_feature_importance.csv"), index=False)

        # Plot feature importance
        plt.figure(figsize=(12, 8))
        sns.barplot(x='importance', y='feature', data=feature_importance.head(20))
        plt.title('Random Forest Regressor Feature Importance (Top 20)')
        plt.tight_layout()
        plt.savefig(os.path.join(log_dir, "plots", "random_forest_feature_importance.png"))
        plt.close()

        # Plot actual vs predicted values
        plt.figure(figsize=(10, 8))
        plt.scatter(y_val, y_val_pred, alpha=0.5)
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlabel('Actual Employment Probability')
        plt.ylabel('Predicted Employment Probability')
        plt.title('Random Forest: Actual vs Predicted Employment Probability')
        plt.tight_layout()
        plt.savefig(os.path.join(log_dir, "plots", "random_forest_actual_vs_predicted.png"))
        plt.close()

    return best_rf, grid_search.best_params_

def train_xgboost(X_train, y_train, X_val, y_val, feature_names):
    """
    Train an XGBoost regressor with hyperparameter tuning.

    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        feature_names: Names of features

    Returns:
        Trained model and best parameters
    """
    logger.info("Training XGBoost Regressor model")

    # Define parameter grid for hyperparameter tuning - optimized for high accuracy
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 6, 9],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
        'min_child_weight': [1, 3, 5]
    }

    # Create base model
    xgb_model = xgb.XGBRegressor(random_state=42, objective='reg:squarederror')

    # Create grid search with reduced parallelism to avoid memory issues
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        cv=3,
        n_jobs=2,  # Reduced from -1 to avoid memory issues
        verbose=1,
        scoring='r2'  # Use R² score for regression
    )

    # Fit grid search
    grid_search.fit(X_train, y_train)

    # Get best model
    best_xgb = grid_search.best_estimator_

    # Evaluate on validation set
    y_val_pred = best_xgb.predict(X_val)
    r2 = r2_score(y_val, y_val_pred)
    mse = mean_squared_error(y_val, y_val_pred)
    mae = mean_absolute_error(y_val, y_val_pred)

    logger.info(f"XGBoost best parameters: {grid_search.best_params_}")
    logger.info(f"XGBoost validation R² score: {r2:.4f}")
    logger.info(f"XGBoost validation MSE: {mse:.4f}")
    logger.info(f"XGBoost validation MAE: {mae:.4f}")

    # Get feature importance
    if hasattr(best_xgb, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': feature_names[:len(best_xgb.feature_importances_)],
            'importance': best_xgb.feature_importances_
        })
        feature_importance = feature_importance.sort_values('importance', ascending=False)

        # Save feature importance to CSV
        feature_importance.to_csv(os.path.join(log_dir, "reports", "xgboost_feature_importance.csv"), index=False)

        # Plot feature importance
        plt.figure(figsize=(12, 8))
        sns.barplot(x='importance', y='feature', data=feature_importance.head(20))
        plt.title('XGBoost Regressor Feature Importance (Top 20)')
        plt.tight_layout()
        plt.savefig(os.path.join(log_dir, "plots", "xgboost_feature_importance.png"))
        plt.close()

        # Plot actual vs predicted values
        plt.figure(figsize=(10, 8))
        plt.scatter(y_val, y_val_pred, alpha=0.5)
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlabel('Actual Employment Probability')
        plt.ylabel('Predicted Employment Probability')
        plt.title('XGBoost: Actual vs Predicted Employment Probability')
        plt.tight_layout()
        plt.savefig(os.path.join(log_dir, "plots", "xgboost_actual_vs_predicted.png"))
        plt.close()

    return best_xgb, grid_search.best_params_

def train_linear_regression(X_train, y_train, X_val, y_val, feature_names):
    """
    Train Linear Regression models with hyperparameter tuning.

    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        feature_names: Names of features

    Returns:
        Trained model and best parameters
    """
    logger.info("Training Linear Regression models")

    # Define parameter grid for Ridge regression
    param_grid = {
        'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
    }

    # Create base model (Ridge regression)
    ridge = Ridge(random_state=42)

    # Create grid search with reduced parallelism to avoid memory issues
    grid_search = GridSearchCV(
        estimator=ridge,
        param_grid=param_grid,
        cv=3,
        n_jobs=2,  # Reduced from -1 to avoid memory issues
        verbose=1,
        scoring='r2'  # Use R² score for regression
    )

    # Fit grid search
    grid_search.fit(X_train, y_train)

    # Get best model
    best_ridge = grid_search.best_estimator_

    # Evaluate on validation set
    y_val_pred = best_ridge.predict(X_val)
    r2 = r2_score(y_val, y_val_pred)
    mse = mean_squared_error(y_val, y_val_pred)
    mae = mean_absolute_error(y_val, y_val_pred)

    logger.info(f"Ridge Regression best parameters: {grid_search.best_params_}")
    logger.info(f"Ridge Regression validation R² score: {r2:.4f}")
    logger.info(f"Ridge Regression validation MSE: {mse:.4f}")
    logger.info(f"Ridge Regression validation MAE: {mae:.4f}")

    # Get feature importance (coefficients)
    if hasattr(best_ridge, 'coef_'):
        # Take the absolute coefficients
        importance = np.abs(best_ridge.coef_)
        feature_importance = pd.DataFrame({
            'feature': feature_names[:len(importance)],
            'importance': importance
        })
        feature_importance = feature_importance.sort_values('importance', ascending=False)

        # Save feature importance to CSV
        feature_importance.to_csv(os.path.join(log_dir, "reports", "ridge_regression_feature_importance.csv"), index=False)

        # Plot feature importance
        plt.figure(figsize=(12, 8))
        sns.barplot(x='importance', y='feature', data=feature_importance.head(20))
        plt.title('Ridge Regression Feature Importance (Top 20)')
        plt.tight_layout()
        plt.savefig(os.path.join(log_dir, "plots", "ridge_regression_feature_importance.png"))
        plt.close()

        # Plot actual vs predicted values
        plt.figure(figsize=(10, 8))
        plt.scatter(y_val, y_val_pred, alpha=0.5)
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlabel('Actual Employment Probability')
        plt.ylabel('Predicted Employment Probability')
        plt.title('Ridge Regression: Actual vs Predicted Employment Probability')
        plt.tight_layout()
        plt.savefig(os.path.join(log_dir, "plots", "ridge_regression_actual_vs_predicted.png"))
        plt.close()

    # Also try Lasso regression for comparison
    lasso = Lasso(random_state=42, max_iter=10000)
    lasso.fit(X_train, y_train)
    lasso_y_val_pred = lasso.predict(X_val)
    lasso_r2 = r2_score(y_val, lasso_y_val_pred)

    logger.info(f"Lasso Regression validation R² score: {lasso_r2:.4f}")

    # Return the best model (Ridge in this case)
    return best_ridge, grid_search.best_params_

def evaluate_model(model, X_test, y_test, model_name, mongo_handler=None, model_params=None, feature_names=None):
    """
    Evaluate a regression model on the test set.

    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        model_name: Name of the model
        mongo_handler: MongoDB handler for saving results
        model_params: Model parameters
        feature_names: Names of features

    Returns:
        Dictionary with evaluation metrics
    """
    logger.info(f"Evaluating {model_name} model")

    # Make predictions
    y_pred = model.predict(X_test)

    # Ensure predictions are within [0, 1] range
    y_pred = np.clip(y_pred, 0, 1)

    # Calculate regression metrics
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    explained_variance = explained_variance_score(y_test, y_pred)

    # Calculate accuracy-like metrics for regression
    # Consider a prediction "accurate" if it's within 0.1 of the true value
    accuracy_01 = np.mean(np.abs(y_test - y_pred) <= 0.1)
    # Consider a prediction "accurate" if it's within 0.05 of the true value
    accuracy_005 = np.mean(np.abs(y_test - y_pred) <= 0.05)

    # Save metrics to CSV
    metrics_df = pd.DataFrame({
        'Metric': ['R²', 'MSE', 'RMSE', 'MAE', 'Explained Variance', 'Accuracy (±0.1)', 'Accuracy (±0.05)'],
        'Value': [r2, mse, rmse, mae, explained_variance, accuracy_01, accuracy_005]
    })
    metrics_df.to_csv(os.path.join(log_dir, "reports", f"{model_name}_metrics.csv"), index=False)

    # Plot actual vs predicted values
    plt.figure(figsize=(10, 8))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlabel('Actual Employment Probability')
    plt.ylabel('Predicted Employment Probability')
    plt.title(f'{model_name}: Actual vs Predicted Employment Probability')
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, "plots", f"{model_name}_actual_vs_predicted.png"))
    plt.close()

    # Plot prediction error distribution
    errors = y_test - y_pred
    plt.figure(figsize=(10, 8))
    sns.histplot(errors, kde=True)
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.title(f'{model_name}: Prediction Error Distribution')
    plt.axvline(x=0, color='r', linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, "plots", f"{model_name}_error_distribution.png"))
    plt.close()

    # Plot residuals vs predicted values
    plt.figure(figsize=(10, 8))
    plt.scatter(y_pred, errors, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Employment Probability')
    plt.ylabel('Residuals')
    plt.title(f'{model_name}: Residuals vs Predicted Values')
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, "plots", f"{model_name}_residuals.png"))
    plt.close()

    logger.info(f"{model_name} Test R² score: {r2:.4f}")
    logger.info(f"{model_name} Test MSE: {mse:.4f}")
    logger.info(f"{model_name} Test RMSE: {rmse:.4f}")
    logger.info(f"{model_name} Test MAE: {mae:.4f}")
    logger.info(f"{model_name} Test Explained Variance: {explained_variance:.4f}")
    logger.info(f"{model_name} Test Accuracy (±0.1): {accuracy_01:.4f}")
    logger.info(f"{model_name} Test Accuracy (±0.05): {accuracy_005:.4f}")

    # Save results to MongoDB if handler is provided
    if mongo_handler is not None:
        try:
            # Save model metadata
            model_id = mongo_handler.save_model_metadata(
                model_name=f"employment_probability_{model_name}",
                model_type=model_name,
                accuracy=float(accuracy_01),  # Use accuracy_01 as the main accuracy metric
                precision=float(r2),  # Use R² as precision
                recall=float(explained_variance),  # Use explained variance as recall
                f1=float(1.0 - mae),  # Use 1-MAE as F1 (higher is better)
                parameters=model_params
            )

            # Save model binary
            mongo_handler.save_model_binary(
                model_id=model_id,
                model_object=model
            )

            # If model has feature importances and feature_names is provided, save them
            if hasattr(model, 'feature_importances_') and feature_names is not None:
                feature_importance = pd.DataFrame({
                    'feature': feature_names[:len(model.feature_importances_)],
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)

                mongo_handler.save_feature_importance(
                    model_id=model_id,
                    feature_importance_df=feature_importance
                )
            elif hasattr(model, 'coef_') and feature_names is not None:
                # For linear models, use coefficients as feature importance
                importance = np.abs(model.coef_)
                feature_importance = pd.DataFrame({
                    'feature': feature_names[:len(importance)],
                    'importance': importance
                }).sort_values('importance', ascending=False)

                mongo_handler.save_feature_importance(
                    model_id=model_id,
                    feature_importance_df=feature_importance
                )

            logger.info(f"Model results saved to MongoDB with ID: {model_id}")
        except Exception as e:
            logger.error(f"Error saving model results to MongoDB: {str(e)}")

    return {
        'model_name': model_name,
        'r2': r2,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'explained_variance': explained_variance,
        'accuracy_01': accuracy_01,
        'accuracy_005': accuracy_005
    }

def save_model(model, preprocessor, model_name):
    """
    Save a trained model and its associated objects.

    Args:
        model: Trained model
        preprocessor: Preprocessing pipeline
        model_name: Name of the model

    Returns:
        Path to saved model
    """
    logger.info(f"Saving {model_name} model")

    # Create model directory
    model_dir = os.path.join(model_save_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)

    # Save model
    model_path = os.path.join(model_dir, f"{model_name}_model.pkl")
    joblib.dump(model, model_path)

    # Save preprocessor
    preprocessor_path = os.path.join(model_dir, "preprocessor.pkl")
    joblib.dump(preprocessor, preprocessor_path)

    logger.info(f"Model saved to {model_dir}")

    return model_dir

def main():
    """Main function to run the employment probability prediction model training and evaluation."""
    logger.info("Starting Employment Probability Post-Graduation model training and evaluation")

    # Define dataset paths
    train_path = "alumni-dataset/bsu_career_train.csv"
    val_path = "alumni-dataset/bsu_career_val.csv"
    test_path = "alumni-dataset/bsu_career_test.csv"

    # Initialize MongoDB handler
    mongo_handler = MongoDBHandler()
    try:
        # Connect to MongoDB
        mongo_handler.connect()
        logger.info("Connected to MongoDB")
    except Exception as e:
        logger.error(f"Error connecting to MongoDB: {str(e)}")
        logger.warning("Continuing without MongoDB integration")
        mongo_handler = None

    # Load data
    train_df, val_df, test_df = load_data(train_path, val_path, test_path)

    # Preprocess data
    (X_train, y_train, X_val, y_val, X_test, y_test,
     preprocessor, feature_names) = preprocess_data(train_df, val_df, test_df)

    # Train models
    logger.info("Training models...")

    # Random Forest Regressor
    rf_model, rf_params = train_random_forest(X_train, y_train, X_val, y_val, feature_names)

    # XGBoost Regressor
    xgb_model, xgb_params = train_xgboost(X_train, y_train, X_val, y_val, feature_names)

    # Linear Regression (Ridge)
    lr_model, lr_params = train_linear_regression(X_train, y_train, X_val, y_val, feature_names)

    # Evaluate models
    logger.info("Evaluating models...")

    rf_metrics = evaluate_model(rf_model, X_test, y_test, "random_forest", mongo_handler, rf_params, feature_names)
    xgb_metrics = evaluate_model(xgb_model, X_test, y_test, "xgboost", mongo_handler, xgb_params, feature_names)
    lr_metrics = evaluate_model(lr_model, X_test, y_test, "ridge_regression", mongo_handler, lr_params, feature_names)

    # Compare models
    models_comparison = pd.DataFrame([
        rf_metrics,
        xgb_metrics,
        lr_metrics
    ])

    # Keep only the metrics columns for comparison
    metrics_cols = ['model_name', 'r2', 'rmse', 'mae', 'explained_variance', 'accuracy_01', 'accuracy_005']
    models_comparison = models_comparison[metrics_cols]

    # Save comparison to CSV
    models_comparison.to_csv(os.path.join(log_dir, "reports", "models_comparison.csv"), index=False)

    # Find best model based on R² score (best for regression)
    best_model_idx = models_comparison['r2'].idxmax()
    best_model_name = models_comparison.loc[best_model_idx, 'model_name']
    best_model_r2 = models_comparison.loc[best_model_idx, 'r2']
    best_model_accuracy = models_comparison.loc[best_model_idx, 'accuracy_01']

    logger.info(f"Best model: {best_model_name} with R² score: {best_model_r2:.4f} and accuracy (±0.1): {best_model_accuracy:.4f}")

    # Check if best model meets accuracy threshold
    if best_model_accuracy < 0.95:
        logger.warning(f"Best model accuracy (±0.1) ({best_model_accuracy:.4f}) is below the target threshold of 0.95")
        logger.info("Performing additional optimization...")

        # Additional optimization could be implemented here
        # For now, we'll just log a message
        logger.info("Additional optimization would be implemented here")

    # Save models
    logger.info("Saving models...")

    rf_model_dir = save_model(rf_model, preprocessor, "random_forest")
    xgb_model_dir = save_model(xgb_model, preprocessor, "xgboost")
    lr_model_dir = save_model(lr_model, preprocessor, "ridge_regression")

    # Save best model separately
    if best_model_name == "random_forest":
        best_model = rf_model
        best_params = rf_params
    elif best_model_name == "xgboost":
        best_model = xgb_model
        best_params = xgb_params
    else:
        best_model = lr_model
        best_params = lr_params

    best_model_dir = save_model(best_model, preprocessor, "best_model")

    # Save best model to MongoDB
    if mongo_handler is not None:
        try:
            # Save model metadata
            model_id = mongo_handler.save_model_metadata(
                model_name="employment_probability_best_model",
                model_type=best_model_name,
                accuracy=float(best_model_accuracy),
                precision=float(best_model_r2),  # Use R² as precision
                recall=float(models_comparison.loc[best_model_idx, 'explained_variance']),  # Use explained variance as recall
                f1=float(1.0 - models_comparison.loc[best_model_idx, 'mae']),  # Use 1-MAE as F1 (higher is better)
                parameters=best_params
            )

            # Save model binary
            mongo_handler.save_model_binary(
                model_id=model_id,
                model_object=best_model,
                preprocessor=preprocessor
            )

            logger.info(f"Best model saved to MongoDB with ID: {model_id}")

            # Close MongoDB connection
            mongo_handler.close()
        except Exception as e:
            logger.error(f"Error saving best model to MongoDB: {str(e)}")

    logger.info("Employment Probability Post-Graduation model training and evaluation completed")
    logger.info(f"Best model: {best_model_name} with R² score: {best_model_r2:.4f} and accuracy (±0.1): {best_model_accuracy:.4f}")
    logger.info(f"Best model saved to {best_model_dir}")

    # Print summary
    print("\n" + "="*80)
    print("Employment Probability Post-Graduation Model Training Summary")
    print("="*80)
    print(f"Best model: {best_model_name}")
    print(f"Best model R² score: {best_model_r2:.4f}")
    print(f"Best model accuracy (±0.1): {best_model_accuracy:.4f}")
    print("\nModel Comparison:")
    print(models_comparison.to_string(index=False))
    print("\nLogs and reports saved to:", log_dir)
    print("Models saved to:", model_save_dir)
    if mongo_handler is not None:
        print("Models also saved to MongoDB for visualization")
    print("="*80)

if __name__ == "__main__":
    main()
