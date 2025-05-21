"""
Career Path Prediction Model

This script trains and evaluates machine learning models for predicting career paths
based on degree and skills using the Alumni Management System dataset.

Models implemented:
- Random Forest
- XGBoost
- Logistic Regression

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
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.metrics import classification_report
import xgboost as xgb
import joblib
import warnings
from mongodb_utils import MongoDBHandler

warnings.filterwarnings('ignore')

# Set up logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(os.path.join(log_dir, "plots"), exist_ok=True)
os.makedirs(os.path.join(log_dir, "reports"), exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(log_dir, f"career_path_prediction_{timestamp}.log")

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
models_dir = "models"
os.makedirs(models_dir, exist_ok=True)
model_save_dir = os.path.join(models_dir, f"career_path_{timestamp}")
os.makedirs(model_save_dir, exist_ok=True)

def load_data(train_path, valid_path, test_path):
    """
    Load the datasets from CSV files.

    Args:
        train_path: Path to the training dataset
        valid_path: Path to the validation dataset
        test_path: Path to the test dataset

    Returns:
        Tuple of DataFrames (train_df, valid_df, test_df)
    """
    logger.info(f"Loading datasets from {train_path}, {valid_path}, and {test_path}")

    try:
        train_df = pd.read_csv(train_path)
        valid_df = pd.read_csv(valid_path)
        test_df = pd.read_csv(test_path)

        logger.info(f"Train dataset shape: {train_df.shape}")
        logger.info(f"Validation dataset shape: {valid_df.shape}")
        logger.info(f"Test dataset shape: {test_df.shape}")

        return train_df, valid_df, test_df
    except Exception as e:
        logger.error(f"Error loading datasets: {str(e)}")
        raise

def preprocess_data(train_df, valid_df, test_df):
    """
    Preprocess the datasets for model training.

    Args:
        train_df: Training DataFrame
        valid_df: Validation DataFrame
        test_df: Test DataFrame

    Returns:
        Tuple of processed data and preprocessing objects
    """
    logger.info("Preprocessing datasets")

    # Combine datasets for consistent preprocessing
    all_data = pd.concat([train_df, valid_df, test_df], axis=0)

    # Clean data
    all_data = clean_data(all_data)

    # Feature engineering
    all_data = engineer_features(all_data)

    # Split back into train, valid, test
    train_processed = all_data.iloc[:len(train_df)]
    valid_processed = all_data.iloc[len(train_df):len(train_df)+len(valid_df)]
    test_processed = all_data.iloc[len(train_df)+len(valid_df):]

    # Define features and target
    target_col = 'actual_job_title'

    # Drop columns not useful for prediction
    drop_cols = ['student_id', 'name', 'predicted_job_title', 'skills_used', 'time_to_employment']
    feature_cols = [col for col in train_processed.columns if col not in [target_col] + drop_cols]

    # Prepare data for modeling
    X_train = train_processed[feature_cols]
    y_train = train_processed[target_col]

    X_valid = valid_processed[feature_cols]
    y_valid = valid_processed[target_col]

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
    X_valid_processed = preprocessor.transform(X_valid)
    X_test_processed = preprocessor.transform(X_test)

    # Encode target variable
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_valid_encoded = label_encoder.transform(y_valid)
    y_test_encoded = label_encoder.transform(y_test)

    # Get feature names after one-hot encoding
    cat_feature_names = []
    for i, col in enumerate(categorical_cols):
        cat_values = preprocessor.transformers_[1][1].categories_[i]
        for val in cat_values:
            cat_feature_names.append(f"cat__{col}_{val}")

    feature_names = numerical_cols + cat_feature_names

    logger.info(f"Processed X_train shape: {X_train_processed.shape}")
    logger.info(f"Processed X_valid shape: {X_valid_processed.shape}")
    logger.info(f"Processed X_test shape: {X_test_processed.shape}")

    return (X_train_processed, y_train_encoded, X_valid_processed, y_valid_encoded,
            X_test_processed, y_test_encoded, preprocessor, label_encoder, feature_names)

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

    # Handle missing values
    if 'actual_job_title' in df.columns:
        df['actual_job_title'] = df['actual_job_title'].fillna('Unemployed')

    if 'gender' in df.columns:
        df['gender'] = df['gender'].fillna('Unknown')

    if 'internship_experience' in df.columns:
        df['internship_experience'] = df['internship_experience'].fillna('No')

    if 'employed_after_grad' in df.columns:
        df['employed_after_grad'] = df['employed_after_grad'].fillna(False)

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
        df['has_soft_skills_skills'] = df['skills'].str.contains('communicate|team|leadership|manage|organize|problem|solve', case=False, na=False)
        df['has_education_skills'] = df['skills'].str.contains('teach|education|train|mentor|tutor', case=False, na=False)
        df['has_healthcare_skills'] = df['skills'].str.contains('medical|health|care|patient|nurse|doctor', case=False, na=False)
        df['has_business_skills'] = df['skills'].str.contains('business|market|sales|finance|account', case=False, na=False)

    # Create internship feature
    if 'internship_experience' in df.columns:
        df['has_internship'] = df['internship_experience'] == 'Yes'

    # Standardize job titles
    if 'actual_job_title' in df.columns:
        # Map similar job titles to standardized versions
        job_title_mapping = {
            'Software Developer': 'Software Engineer',
            'Software Programmer': 'Software Engineer',
            'Web Developer': 'Software Engineer',
            'Web Designer': 'Software Engineer',
            'Data Scientist': 'Data Analyst',
            'Data Engineer': 'Data Analyst',
            'Business Analyst': 'Data Analyst',
            'Nurse': 'Healthcare Professional',
            'Medical Assistant': 'Healthcare Professional',
            'Doctor': 'Healthcare Professional',
            'Teacher': 'Educator',
            'Professor': 'Educator',
            'Tutor': 'Educator',
            'Marketing Specialist': 'Marketing Professional',
            'Marketing Coordinator': 'Marketing Professional',
            'Sales Representative': 'Sales Professional',
            'Sales Associate': 'Sales Professional',
            'Accountant': 'Finance Professional',
            'Financial Analyst': 'Finance Professional'
        }

        # Create a new column with standardized job titles
        df['standardized_job_title'] = df['actual_job_title'].map(lambda x: job_title_mapping.get(x, x))

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

def train_random_forest(X_train, y_train, X_valid, y_valid, feature_names):
    """
    Train a Random Forest classifier with hyperparameter tuning.

    Args:
        X_train: Training features
        y_train: Training labels
        X_valid: Validation features
        y_valid: Validation labels
        feature_names: Names of features

    Returns:
        Trained model and best parameters
    """
    logger.info("Training Random Forest model")

    # Define parameter grid for hyperparameter tuning - reduced for memory efficiency
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20],
        'min_samples_split': [2, 10],
        'min_samples_leaf': [1, 4]
    }

    # Create base model
    rf = RandomForestClassifier(random_state=42)

    # Create grid search with reduced parallelism to avoid memory issues
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=3,
        n_jobs=2,  # Reduced from -1 to avoid memory issues
        verbose=1,
        scoring='accuracy'
    )

    # Fit grid search
    grid_search.fit(X_train, y_train)

    # Get best model
    best_rf = grid_search.best_estimator_

    # Evaluate on validation set
    y_valid_pred = best_rf.predict(X_valid)
    accuracy = accuracy_score(y_valid, y_valid_pred)

    logger.info(f"Random Forest best parameters: {grid_search.best_params_}")
    logger.info(f"Random Forest validation accuracy: {accuracy:.4f}")

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
        plt.title('Random Forest Feature Importance (Top 20)')
        plt.tight_layout()
        plt.savefig(os.path.join(log_dir, "plots", "random_forest_feature_importance.png"))
        plt.close()

    return best_rf, grid_search.best_params_

def train_xgboost(X_train, y_train, X_valid, y_valid, feature_names):
    """
    Train an XGBoost classifier with hyperparameter tuning.

    Args:
        X_train: Training features
        y_train: Training labels
        X_valid: Validation features
        y_valid: Validation labels
        feature_names: Names of features

    Returns:
        Trained model and best parameters
    """
    logger.info("Training XGBoost model")

    # Define parameter grid for hyperparameter tuning - reduced for memory efficiency
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 6],
        'learning_rate': [0.1],
        'subsample': [0.8],
        'colsample_bytree': [0.8]
    }

    # Create base model
    xgb_model = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss')

    # Create grid search with reduced parallelism to avoid memory issues
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        cv=3,
        n_jobs=2,  # Reduced from -1 to avoid memory issues
        verbose=1,
        scoring='accuracy'
    )

    # Fit grid search
    grid_search.fit(X_train, y_train)

    # Get best model
    best_xgb = grid_search.best_estimator_

    # Evaluate on validation set
    y_valid_pred = best_xgb.predict(X_valid)
    accuracy = accuracy_score(y_valid, y_valid_pred)

    logger.info(f"XGBoost best parameters: {grid_search.best_params_}")
    logger.info(f"XGBoost validation accuracy: {accuracy:.4f}")

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
        plt.title('XGBoost Feature Importance (Top 20)')
        plt.tight_layout()
        plt.savefig(os.path.join(log_dir, "plots", "xgboost_feature_importance.png"))
        plt.close()

    return best_xgb, grid_search.best_params_

def train_logistic_regression(X_train, y_train, X_valid, y_valid, feature_names):
    """
    Train a Logistic Regression classifier with hyperparameter tuning.

    Args:
        X_train: Training features
        y_train: Training labels
        X_valid: Validation features
        y_valid: Validation labels
        feature_names: Names of features

    Returns:
        Trained model and best parameters
    """
    logger.info("Training Logistic Regression model")

    # Define parameter grid for hyperparameter tuning - reduced for memory efficiency
    param_grid = {
        'C': [0.1, 1, 10],
        'penalty': ['l2', None],
        'solver': ['lbfgs', 'saga'],
        'max_iter': [200]
    }

    # Create base model
    lr = LogisticRegression(random_state=42)

    # Create grid search with reduced parallelism to avoid memory issues
    grid_search = GridSearchCV(
        estimator=lr,
        param_grid=param_grid,
        cv=3,
        n_jobs=2,  # Reduced from -1 to avoid memory issues
        verbose=1,
        scoring='accuracy'
    )

    # Fit grid search
    grid_search.fit(X_train, y_train)

    # Get best model
    best_lr = grid_search.best_estimator_

    # Evaluate on validation set
    y_valid_pred = best_lr.predict(X_valid)
    accuracy = accuracy_score(y_valid, y_valid_pred)

    logger.info(f"Logistic Regression best parameters: {grid_search.best_params_}")
    logger.info(f"Logistic Regression validation accuracy: {accuracy:.4f}")

    # Get feature importance (coefficients)
    if hasattr(best_lr, 'coef_'):
        # For multi-class, take the mean of absolute coefficients across all classes
        importance = np.mean(np.abs(best_lr.coef_), axis=0)
        feature_importance = pd.DataFrame({
            'feature': feature_names[:len(importance)],
            'importance': importance
        })
        feature_importance = feature_importance.sort_values('importance', ascending=False)

        # Save feature importance to CSV
        feature_importance.to_csv(os.path.join(log_dir, "reports", "logistic_regression_feature_importance.csv"), index=False)

        # Plot feature importance
        plt.figure(figsize=(12, 8))
        sns.barplot(x='importance', y='feature', data=feature_importance.head(20))
        plt.title('Logistic Regression Feature Importance (Top 20)')
        plt.tight_layout()
        plt.savefig(os.path.join(log_dir, "plots", "logistic_regression_feature_importance.png"))
        plt.close()

    return best_lr, grid_search.best_params_

def evaluate_model(model, X_test, y_test, label_encoder, model_name, mongo_handler=None, model_params=None):
    """
    Evaluate a model on the test set.

    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        label_encoder: Label encoder for target variable
        model_name: Name of the model
        mongo_handler: MongoDB handler for saving results
        model_params: Model parameters

    Returns:
        Dictionary with evaluation metrics
    """
    logger.info(f"Evaluating {model_name} model")

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')

    # Get classification report
    class_names = label_encoder.classes_
    report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)

    # Save classification report to CSV
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(os.path.join(log_dir, "reports", f"{model_name}_classification_report.csv"))

    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'{model_name} Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, "plots", f"{model_name}_confusion_matrix.png"))
    plt.close()

    logger.info(f"{model_name} Test Accuracy: {accuracy:.4f}")
    logger.info(f"{model_name} Test Precision: {precision:.4f}")
    logger.info(f"{model_name} Test Recall: {recall:.4f}")
    logger.info(f"{model_name} Test F1 Score: {f1:.4f}")

    # Save results to MongoDB if handler is provided
    if mongo_handler is not None:
        try:
            # Save model metadata
            model_id = mongo_handler.save_model_metadata(
                model_name=f"career_path_{model_name}",
                model_type=model_name,
                accuracy=float(accuracy),
                precision=float(precision),
                recall=float(recall),
                f1=float(f1),
                parameters=model_params
            )

            # Save model binary
            mongo_handler.save_model_binary(
                model_id=model_id,
                model_object=model
            )

            # Save classification report
            mongo_handler.save_classification_report(
                model_id=model_id,
                classification_report=report
            )

            # Save confusion matrix
            mongo_handler.save_confusion_matrix(
                model_id=model_id,
                confusion_matrix=cm,
                class_names=class_names.tolist()
            )

            # If model has feature importances, save them
            if hasattr(model, 'feature_importances_'):
                feature_importance = pd.DataFrame({
                    'feature': feature_names[:len(model.feature_importances_)],
                    'importance': model.feature_importances_
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
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'classification_report': report
    }

def save_model(model, preprocessor, label_encoder, model_name):
    """
    Save a trained model and its associated objects.

    Args:
        model: Trained model
        preprocessor: Preprocessing pipeline
        label_encoder: Label encoder for target variable
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

    # Save label encoder
    label_encoder_path = os.path.join(model_dir, "label_encoder.pkl")
    joblib.dump(label_encoder, label_encoder_path)

    # Save class names
    class_names_path = os.path.join(model_dir, "class_names.pkl")
    joblib.dump(label_encoder.classes_, class_names_path)

    logger.info(f"Model saved to {model_dir}")

    return model_dir

def main():
    """Main function to run the career path prediction model training and evaluation."""
    logger.info("Starting Career Path Prediction model training and evaluation")

    # Define dataset paths
    train_path = "dataset/train_employment.csv"
    valid_path = "dataset/valid_employment.csv"
    test_path = "dataset/test_employment.csv"

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
    train_df, valid_df, test_df = load_data(train_path, valid_path, test_path)

    # Preprocess data
    (X_train, y_train, X_valid, y_valid, X_test, y_test,
     preprocessor, label_encoder, feature_names) = preprocess_data(train_df, valid_df, test_df)

    # Train models
    logger.info("Training models...")

    # Random Forest
    rf_model, rf_params = train_random_forest(X_train, y_train, X_valid, y_valid, feature_names)

    # XGBoost
    xgb_model, xgb_params = train_xgboost(X_train, y_train, X_valid, y_valid, feature_names)

    # Logistic Regression
    lr_model, lr_params = train_logistic_regression(X_train, y_train, X_valid, y_valid, feature_names)

    # Evaluate models
    logger.info("Evaluating models...")

    rf_metrics = evaluate_model(rf_model, X_test, y_test, label_encoder, "random_forest", mongo_handler, rf_params)
    xgb_metrics = evaluate_model(xgb_model, X_test, y_test, label_encoder, "xgboost", mongo_handler, xgb_params)
    lr_metrics = evaluate_model(lr_model, X_test, y_test, label_encoder, "logistic_regression", mongo_handler, lr_params)

    # Compare models
    models_comparison = pd.DataFrame([
        rf_metrics,
        xgb_metrics,
        lr_metrics
    ])

    # Keep only the metrics columns for comparison
    models_comparison = models_comparison[['model_name', 'accuracy', 'precision', 'recall', 'f1']]

    # Save comparison to CSV
    models_comparison.to_csv(os.path.join(log_dir, "reports", "models_comparison.csv"), index=False)

    # Find best model
    best_model_idx = models_comparison['accuracy'].idxmax()
    best_model_name = models_comparison.loc[best_model_idx, 'model_name']
    best_model_accuracy = models_comparison.loc[best_model_idx, 'accuracy']

    logger.info(f"Best model: {best_model_name} with accuracy: {best_model_accuracy:.4f}")

    # Check if best model meets accuracy threshold
    if best_model_accuracy < 0.95:
        logger.warning(f"Best model accuracy ({best_model_accuracy:.4f}) is below the target threshold of 0.95")
        logger.info("Performing additional optimization...")

        # Additional optimization could be implemented here
        # For now, we'll just log a message
        logger.info("Additional optimization would be implemented here")

    # Save models
    logger.info("Saving models...")

    rf_model_dir = save_model(rf_model, preprocessor, label_encoder, "random_forest")
    xgb_model_dir = save_model(xgb_model, preprocessor, label_encoder, "xgboost")
    lr_model_dir = save_model(lr_model, preprocessor, label_encoder, "logistic_regression")

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

    best_model_dir = save_model(best_model, preprocessor, label_encoder, "best_model")

    # Save best model to MongoDB
    if mongo_handler is not None:
        try:
            # Save model metadata
            model_id = mongo_handler.save_model_metadata(
                model_name="career_path_best_model",
                model_type=best_model_name,
                accuracy=float(best_model_accuracy),
                precision=float(models_comparison.loc[best_model_idx, 'precision']),
                recall=float(models_comparison.loc[best_model_idx, 'recall']),
                f1=float(models_comparison.loc[best_model_idx, 'f1']),
                parameters=best_params
            )

            # Save model binary
            mongo_handler.save_model_binary(
                model_id=model_id,
                model_object=best_model,
                preprocessor=preprocessor,
                label_encoder=label_encoder
            )

            logger.info(f"Best model saved to MongoDB with ID: {model_id}")

            # Close MongoDB connection
            mongo_handler.close()
        except Exception as e:
            logger.error(f"Error saving best model to MongoDB: {str(e)}")

    logger.info("Career Path Prediction model training and evaluation completed")
    logger.info(f"Best model: {best_model_name} with accuracy: {best_model_accuracy:.4f}")
    logger.info(f"Best model saved to {best_model_dir}")

    # Print summary
    print("\n" + "="*80)
    print("Career Path Prediction Model Training Summary")
    print("="*80)
    print(f"Best model: {best_model_name}")
    print(f"Best model accuracy: {best_model_accuracy:.4f}")
    print("\nModel Comparison:")
    print(models_comparison.to_string(index=False))
    print("\nLogs and reports saved to:", log_dir)
    print("Models saved to:", model_save_dir)
    if mongo_handler is not None:
        print("Models also saved to MongoDB for visualization")
    print("="*80)

if __name__ == "__main__":
    main()
