"""
Visualize Career Path Prediction Model Results from MongoDB

This script retrieves and visualizes the results of the Career Path Prediction models
stored in MongoDB.

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
from mongodb_utils import MongoDBHandler
import pickle
from bson.objectid import ObjectId

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# Create visualization directory
viz_dir = "visualizations"
os.makedirs(viz_dir, exist_ok=True)

def get_model_metadata(mongo_handler, model_type=None, limit=10):
    """
    Get model metadata from MongoDB.
    
    Args:
        mongo_handler: MongoDB handler
        model_type: Type of model to filter by (e.g., random_forest, xgboost)
        limit: Maximum number of models to retrieve
        
    Returns:
        DataFrame with model metadata
    """
    logger.info(f"Getting model metadata from MongoDB")
    
    collection = mongo_handler.db["model_metadata"]
    
    # Build query
    query = {}
    if model_type:
        query["model_type"] = model_type
    
    # Get data
    cursor = collection.find(query).sort("timestamp", -1).limit(limit)
    
    # Convert to DataFrame
    df = pd.DataFrame(list(cursor))
    
    if df.empty:
        logger.warning("No model metadata found in MongoDB")
        return None
    
    logger.info(f"Retrieved {len(df)} model metadata records")
    return df

def get_feature_importance(mongo_handler, model_id):
    """
    Get feature importance data from MongoDB.
    
    Args:
        mongo_handler: MongoDB handler
        model_id: ID of the model
        
    Returns:
        DataFrame with feature importance data
    """
    logger.info(f"Getting feature importance for model {model_id}")
    
    collection = mongo_handler.db["feature_importance"]
    
    # Get data
    document = collection.find_one({"model_id": model_id})
    
    if not document:
        logger.warning(f"No feature importance found for model {model_id}")
        return None
    
    # Convert to DataFrame
    df = pd.DataFrame(document["feature_importance"])
    
    logger.info(f"Retrieved feature importance with {len(df)} features")
    return df

def get_confusion_matrix(mongo_handler, model_id):
    """
    Get confusion matrix data from MongoDB.
    
    Args:
        mongo_handler: MongoDB handler
        model_id: ID of the model
        
    Returns:
        Tuple of (confusion_matrix, class_names)
    """
    logger.info(f"Getting confusion matrix for model {model_id}")
    
    collection = mongo_handler.db["confusion_matrices"]
    
    # Get data
    document = collection.find_one({"model_id": model_id})
    
    if not document:
        logger.warning(f"No confusion matrix found for model {model_id}")
        return None, None
    
    # Convert to numpy array
    cm = np.array(document["confusion_matrix"])
    class_names = document["class_names"]
    
    logger.info(f"Retrieved confusion matrix with shape {cm.shape}")
    return cm, class_names

def visualize_model_comparison(metadata_df):
    """
    Visualize model comparison.
    
    Args:
        metadata_df: DataFrame with model metadata
    """
    logger.info("Visualizing model comparison")
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Plot accuracy
    plt.subplot(2, 2, 1)
    sns.barplot(x="model_type", y="accuracy", data=metadata_df)
    plt.title("Model Accuracy")
    plt.ylim(0, 1)
    
    # Plot precision
    plt.subplot(2, 2, 2)
    sns.barplot(x="model_type", y="precision", data=metadata_df)
    plt.title("Model Precision")
    plt.ylim(0, 1)
    
    # Plot recall
    plt.subplot(2, 2, 3)
    sns.barplot(x="model_type", y="recall", data=metadata_df)
    plt.title("Model Recall")
    plt.ylim(0, 1)
    
    # Plot F1 score
    plt.subplot(2, 2, 4)
    sns.barplot(x="model_type", y="f1", data=metadata_df)
    plt.title("Model F1 Score")
    plt.ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, "model_comparison.png"))
    plt.close()
    
    logger.info(f"Model comparison visualization saved to {os.path.join(viz_dir, 'model_comparison.png')}")

def visualize_feature_importance(feature_importance_df, model_type, top_n=20):
    """
    Visualize feature importance.
    
    Args:
        feature_importance_df: DataFrame with feature importance data
        model_type: Type of model
        top_n: Number of top features to show
    """
    logger.info(f"Visualizing feature importance for {model_type}")
    
    # Sort by importance
    df = feature_importance_df.sort_values("importance", ascending=False).head(top_n)
    
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # Plot feature importance
    sns.barplot(x="importance", y="feature", data=df)
    plt.title(f"{model_type} Feature Importance (Top {top_n})")
    plt.tight_layout()
    
    plt.savefig(os.path.join(viz_dir, f"{model_type}_feature_importance.png"))
    plt.close()
    
    logger.info(f"Feature importance visualization saved to {os.path.join(viz_dir, f'{model_type}_feature_importance.png')}")

def visualize_confusion_matrix(cm, class_names, model_type):
    """
    Visualize confusion matrix.
    
    Args:
        cm: Confusion matrix as a numpy array
        class_names: Names of the classes
        model_type: Type of model
    """
    logger.info(f"Visualizing confusion matrix for {model_type}")
    
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # Plot confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'{model_type} Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    plt.savefig(os.path.join(viz_dir, f"{model_type}_confusion_matrix.png"))
    plt.close()
    
    logger.info(f"Confusion matrix visualization saved to {os.path.join(viz_dir, f'{model_type}_confusion_matrix.png')}")

def main():
    """Main function to visualize model results from MongoDB."""
    logger.info("Starting model results visualization from MongoDB")
    
    # Initialize MongoDB handler
    mongo_handler = MongoDBHandler()
    try:
        # Connect to MongoDB
        mongo_handler.connect()
        logger.info("Connected to MongoDB")
    except Exception as e:
        logger.error(f"Error connecting to MongoDB: {str(e)}")
        sys.exit(1)
    
    # Get model metadata
    metadata_df = get_model_metadata(mongo_handler)
    
    if metadata_df is None:
        logger.error("No model metadata found in MongoDB")
        mongo_handler.close()
        sys.exit(1)
    
    # Visualize model comparison
    visualize_model_comparison(metadata_df)
    
    # Visualize results for each model type
    model_types = metadata_df["model_type"].unique()
    
    for model_type in model_types:
        # Get latest model of this type
        model_data = metadata_df[metadata_df["model_type"] == model_type].iloc[0]
        model_id = model_data["_id"]
        
        # Get feature importance
        feature_importance_df = get_feature_importance(mongo_handler, model_id)
        if feature_importance_df is not None:
            visualize_feature_importance(feature_importance_df, model_type)
        
        # Get confusion matrix
        cm, class_names = get_confusion_matrix(mongo_handler, model_id)
        if cm is not None and class_names is not None:
            visualize_confusion_matrix(cm, class_names, model_type)
    
    # Close MongoDB connection
    mongo_handler.close()
    
    logger.info("Model results visualization completed")
    
    # Print summary
    print("\n" + "="*80)
    print("Career Path Prediction Model Visualization Summary")
    print("="*80)
    print(f"Number of models: {len(metadata_df)}")
    print(f"Model types: {', '.join(model_types)}")
    print(f"Visualizations saved to: {viz_dir}")
    print("="*80)

if __name__ == "__main__":
    main()
