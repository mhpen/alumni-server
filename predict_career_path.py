"""
Career Path Prediction - Inference Script

This script demonstrates how to use the trained Career Path Prediction model
to make predictions for new alumni data.

Author: Augment Agent
Date: 2025-05-21
"""

import os
import sys
import logging
import pandas as pd
import joblib
import argparse

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def load_model(model_dir):
    """
    Load the trained model and associated objects.
    
    Args:
        model_dir: Directory containing the model files
        
    Returns:
        Tuple of (model, preprocessor, label_encoder, class_names)
    """
    logger.info(f"Loading model from {model_dir}")
    
    try:
        # Find model file
        model_files = [f for f in os.listdir(model_dir) if f.endswith('_model.pkl')]
        if not model_files:
            raise FileNotFoundError(f"No model file found in {model_dir}")
        
        model_path = os.path.join(model_dir, model_files[0])
        preprocessor_path = os.path.join(model_dir, "preprocessor.pkl")
        label_encoder_path = os.path.join(model_dir, "label_encoder.pkl")
        class_names_path = os.path.join(model_dir, "class_names.pkl")
        
        # Load model and associated objects
        model = joblib.load(model_path)
        preprocessor = joblib.load(preprocessor_path)
        label_encoder = joblib.load(label_encoder_path)
        class_names = joblib.load(class_names_path)
        
        logger.info(f"Model loaded successfully: {model_path}")
        
        return model, preprocessor, label_encoder, class_names
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def predict_career_path(model, preprocessor, label_encoder, data):
    """
    Make career path predictions for new data.
    
    Args:
        model: Trained model
        preprocessor: Preprocessing pipeline
        label_encoder: Label encoder for target variable
        data: DataFrame with new data
        
    Returns:
        DataFrame with predictions
    """
    logger.info("Making predictions")
    
    try:
        # Preprocess data
        X = preprocessor.transform(data)
        
        # Make predictions
        y_pred = model.predict(X)
        y_pred_proba = model.predict_proba(X)
        
        # Convert predictions to class names
        predictions = label_encoder.inverse_transform(y_pred)
        
        # Create results DataFrame
        results = data.copy()
        results['predicted_job_title'] = predictions
        
        # Add prediction probabilities
        for i, class_name in enumerate(label_encoder.classes_):
            results[f'probability_{class_name}'] = y_pred_proba[:, i]
        
        # Add confidence (probability of predicted class)
        results['confidence'] = [proba[pred] for pred, proba in zip(y_pred, y_pred_proba)]
        
        return results
    except Exception as e:
        logger.error(f"Error making predictions: {str(e)}")
        raise

def main():
    """Main function to run the career path prediction inference."""
    parser = argparse.ArgumentParser(description='Career Path Prediction Inference')
    parser.add_argument('--model-dir', type=str, default='models/best_model',
                        help='Directory containing the model files')
    parser.add_argument('--input-file', type=str, required=True,
                        help='CSV file with input data')
    parser.add_argument('--output-file', type=str, default='predictions.csv',
                        help='CSV file to save predictions')
    
    args = parser.parse_args()
    
    try:
        # Load model
        model, preprocessor, label_encoder, class_names = load_model(args.model_dir)
        
        # Load input data
        logger.info(f"Loading input data from {args.input_file}")
        data = pd.read_csv(args.input_file)
        logger.info(f"Input data shape: {data.shape}")
        
        # Make predictions
        results = predict_career_path(model, preprocessor, label_encoder, data)
        
        # Save predictions
        logger.info(f"Saving predictions to {args.output_file}")
        results.to_csv(args.output_file, index=False)
        
        logger.info("Prediction completed successfully")
        
        # Print summary
        print("\n" + "="*80)
        print("Career Path Prediction Summary")
        print("="*80)
        print(f"Input file: {args.input_file}")
        print(f"Number of records: {len(data)}")
        print(f"Predictions saved to: {args.output_file}")
        
        # Print prediction distribution
        pred_counts = results['predicted_job_title'].value_counts()
        print("\nPrediction Distribution:")
        for job_title, count in pred_counts.items():
            print(f"  {job_title}: {count} ({count/len(results)*100:.1f}%)")
        
        print("="*80)
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
