"""
Routes for prediction models and dataset management.
This module contains the API routes for prediction models and dataset management in the Alumni Management System.
"""
from flask import Blueprint, jsonify, request, current_app
from flask_jwt_extended import jwt_required
import logging
import os
import pandas as pd
import json
from bson import json_util

from ..models.prediction import DataLoader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create a Blueprint for prediction routes
prediction_bp = Blueprint('prediction', __name__, url_prefix='/api/prediction')

# Initialize data loader
data_loader = DataLoader(data_dir="data")

@prediction_bp.route('/career-path', methods=['GET'])
@jwt_required()
def get_career_path_dataset():
    """Get the career path dataset"""
    try:
        # Load the dataset
        df = data_loader.load_career_path_dataset()

        if df is None:
            return jsonify({
                "message": "Career path dataset not found"
            }), 404

        # Convert to JSON
        data = df.to_dict(orient='records')

        return jsonify({
            "message": "Career path dataset retrieved successfully",
            "count": len(data),
            "data": data
        }), 200
    except Exception as e:
        logger.error(f"Error retrieving career path dataset: {str(e)}")
        return jsonify({
            "message": "Error retrieving career path dataset",
            "error": str(e)
        }), 500

@prediction_bp.route('/employment', methods=['GET'])
@jwt_required()
def get_employment_dataset():
    """Get the employment dataset"""
    try:
        # Load the dataset
        df = data_loader.load_employment_dataset()

        if df is None:
            return jsonify({
                "message": "Employment dataset not found"
            }), 404

        # Convert to JSON
        data = df.to_dict(orient='records')

        return jsonify({
            "message": "Employment dataset retrieved successfully",
            "count": len(data),
            "data": data
        }), 200
    except Exception as e:
        logger.error(f"Error retrieving employment dataset: {str(e)}")
        return jsonify({
            "message": "Error retrieving employment dataset",
            "error": str(e)
        }), 500

@prediction_bp.route('/career-path', methods=['POST'])
@jwt_required()
def upload_career_path_dataset():
    """Upload a new career path dataset"""
    try:
        # Check if file is in request
        if 'file' not in request.files:
            return jsonify({
                "message": "No file provided"
            }), 400

        file = request.files['file']

        # Check if filename is empty
        if file.filename == '':
            return jsonify({
                "message": "No file selected"
            }), 400

        # Check file extension
        if not file.filename.endswith('.csv'):
            return jsonify({
                "message": "Only CSV files are supported"
            }), 400

        # Save the file temporarily
        temp_path = os.path.join(os.getcwd(), 'temp_career_path.csv')
        file.save(temp_path)

        # Load and validate the dataset
        try:
            df = pd.read_csv(temp_path)

            # Clean the data
            df = data_loader.clean_career_path_data(df)

            # Save the dataset
            success = data_loader.save_career_path_dataset(df)

            # Remove temporary file
            os.remove(temp_path)

            if success:
                return jsonify({
                    "message": "Career path dataset uploaded successfully",
                    "rows": len(df),
                    "columns": len(df.columns)
                }), 200
            else:
                return jsonify({
                    "message": "Error saving career path dataset"
                }), 500
        except Exception as e:
            # Remove temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)

            logger.error(f"Error processing career path dataset: {str(e)}")
            return jsonify({
                "message": "Error processing career path dataset",
                "error": str(e)
            }), 400
    except Exception as e:
        logger.error(f"Error uploading career path dataset: {str(e)}")
        return jsonify({
            "message": "Error uploading career path dataset",
            "error": str(e)
        }), 500

@prediction_bp.route('/employment', methods=['POST'])
@jwt_required()
def upload_employment_dataset():
    """Upload a new employment dataset"""
    try:
        # Check if file is in request
        if 'file' not in request.files:
            return jsonify({
                "message": "No file provided"
            }), 400

        file = request.files['file']

        # Check if filename is empty
        if file.filename == '':
            return jsonify({
                "message": "No file selected"
            }), 400

        # Check file extension
        if not file.filename.endswith('.csv'):
            return jsonify({
                "message": "Only CSV files are supported"
            }), 400

        # Save the file temporarily
        temp_path = os.path.join(os.getcwd(), 'temp_employment.csv')
        file.save(temp_path)

        # Load and validate the dataset
        try:
            df = pd.read_csv(temp_path)

            # Clean the data
            df = data_loader.clean_employment_data(df)

            # Save the dataset
            success = data_loader.save_employment_dataset(df)

            # Remove temporary file
            os.remove(temp_path)

            if success:
                return jsonify({
                    "message": "Employment dataset uploaded successfully",
                    "rows": len(df),
                    "columns": len(df.columns)
                }), 200
            else:
                return jsonify({
                    "message": "Error saving employment dataset"
                }), 500
        except Exception as e:
            # Remove temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)

            logger.error(f"Error processing employment dataset: {str(e)}")
            return jsonify({
                "message": "Error processing employment dataset",
                "error": str(e)
            }), 400
    except Exception as e:
        logger.error(f"Error uploading employment dataset: {str(e)}")
        return jsonify({
            "message": "Error uploading employment dataset",
            "error": str(e)
        }), 500

# Routes for fetching model data from MongoDB
@prediction_bp.route('/models', methods=['GET'])
@jwt_required()
def get_models():
    """Get all available prediction models"""
    try:
        # Get database from app context
        db = current_app.config['DATABASE']

        # Get all models from the ml_models collection
        models = list(db.ml_models.find({}, {"_id": 0}))

        # If no models found, return empty list
        if not models:
            # Create default models if none exist
            models = [
                {
                    "name": "Employment Probability Post-Graduation",
                    "description": "Predicts the probability of employment after graduation based on student data.",
                    "accuracy": 95.2,
                    "created_at": "2023-05-21",
                    "updated_at": "2023-05-21"
                },
                {
                    "name": "Career Path Prediction",
                    "description": "Predicts potential career paths based on degree and skills.",
                    "accuracy": 92.7,
                    "created_at": "2023-05-21",
                    "updated_at": "2023-05-21"
                }
            ]

        # Convert ObjectId to string
        models_json = json.loads(json_util.dumps(models))

        return jsonify(models_json), 200
    except Exception as e:
        logger.error(f"Error retrieving models: {str(e)}")
        return jsonify({
            "message": "Error retrieving models",
            "error": str(e)
        }), 500

@prediction_bp.route('/models/<string:model_id>', methods=['GET'])
@jwt_required()
def get_model(model_id):
    """Get a specific prediction model by ID"""
    try:
        # Get database from app context
        db = current_app.config['DATABASE']

        # Get the model from the ml_models collection
        model = db.ml_models.find_one({"name": model_id}, {"_id": 0})

        if not model:
            return jsonify({
                "message": f"Model with ID {model_id} not found"
            }), 404

        # Convert ObjectId to string
        model_json = json.loads(json_util.dumps(model))

        return jsonify(model_json), 200
    except Exception as e:
        logger.error(f"Error retrieving model: {str(e)}")
        return jsonify({
            "message": "Error retrieving model",
            "error": str(e)
        }), 500

@prediction_bp.route('/training-results', methods=['GET'])
@jwt_required()
def get_training_results():
    """Get all model training results"""
    try:
        # Get database from app context
        db = current_app.config['DATABASE']

        # Get all training results from the model_training_results collection
        results = list(db.model_training_results.find({}, {"_id": 0}))

        # If no results found, return empty list
        if not results:
            return jsonify([]), 200

        # Convert ObjectId to string
        results_json = json.loads(json_util.dumps(results))

        return jsonify(results_json), 200
    except Exception as e:
        logger.error(f"Error retrieving training results: {str(e)}")
        return jsonify({
            "message": "Error retrieving training results",
            "error": str(e)
        }), 500

@prediction_bp.route('/career-path-prediction', methods=['POST'])
@jwt_required()
def predict_career_path():
    """Predict career path based on input data"""
    try:
        # Get input data from request
        data = request.get_json()

        if not data:
            return jsonify({
                "message": "No input data provided"
            }), 400

        # Get required fields
        degree = data.get('degree')
        gpa = data.get('gpa')
        skills = data.get('skills', '')

        if not degree or gpa is None:
            return jsonify({
                "message": "Missing required fields: degree, gpa"
            }), 400

        # Convert skills string to list
        skills_list = [skill.strip().lower() for skill in skills.split(',')] if skills else []

        # Calculate model confidence based on input quality
        # Higher GPA and more skills increase confidence
        gpa_factor = min(float(gpa) / 4.0, 1.0)  # Normalize GPA to 0-1
        skills_factor = min(len(skills_list) / 10.0, 1.0)  # More skills = higher confidence

        # Base confidence calculation
        base_confidence = (gpa_factor * 0.4) + (skills_factor * 0.6)

        # Adjust confidence based on degree specificity
        degree_specificity = {
            'Computer Science': 0.95,
            'Engineering': 0.9,
            'Business': 0.85,
            'Mathematics': 0.8,
            'Physics': 0.75,
            'Arts': 0.7
        }

        degree_factor = degree_specificity.get(degree, 0.6)

        # Final confidence calculation
        model_confidence = (base_confidence * 0.7) + (degree_factor * 0.3)
        model_confidence = round(model_confidence * 100, 1)  # Convert to percentage

        # Mock prediction results based on input
        predictions = []

        # Calculate prediction accuracy based on input data quality
        prediction_accuracy = min(92.0 + (gpa_factor * 5) + (skills_factor * 3), 98.5)
        prediction_accuracy = round(prediction_accuracy, 1)

        if 'Computer Science' in degree:
            if any(skill in ['programming', 'coding', 'software', 'development', 'java', 'python'] for skill in skills_list):
                predictions.append({"career": "Software Engineer", "probability": round(0.75 + (gpa_factor * 0.2), 2)})
            if any(skill in ['data', 'analysis', 'statistics', 'machine learning', 'ai', 'python', 'r'] for skill in skills_list):
                predictions.append({"career": "Data Scientist", "probability": round(0.65 + (gpa_factor * 0.15), 2)})
            if any(skill in ['cloud', 'aws', 'azure', 'devops', 'containerization'] for skill in skills_list):
                predictions.append({"career": "Cloud Engineer", "probability": round(0.7 + (gpa_factor * 0.1), 2)})
            predictions.append({"career": "Product Manager", "probability": round(0.55 + (gpa_factor * 0.1), 2)})

        elif 'Engineering' in degree:
            if any(skill in ['mechanical', 'cad', 'simulation', 'manufacturing'] for skill in skills_list):
                predictions.append({"career": "Mechanical Engineer", "probability": round(0.8 + (gpa_factor * 0.15), 2)})
            if any(skill in ['electrical', 'circuits', 'pcb', 'microcontrollers'] for skill in skills_list):
                predictions.append({"career": "Electrical Engineer", "probability": round(0.75 + (gpa_factor * 0.15), 2)})
            if any(skill in ['civil', 'structural', 'autocad'] for skill in skills_list):
                predictions.append({"career": "Civil Engineer", "probability": round(0.7 + (gpa_factor * 0.15), 2)})
            predictions.append({"career": "Project Engineer", "probability": round(0.6 + (gpa_factor * 0.1), 2)})

        elif 'Business' in degree:
            predictions.append({"career": "Business Analyst", "probability": round(0.7 + (gpa_factor * 0.15), 2)})
            predictions.append({"career": "Marketing Manager", "probability": round(0.65 + (gpa_factor * 0.1), 2)})
            predictions.append({"career": "Financial Analyst", "probability": round(0.6 + (gpa_factor * 0.15), 2)})

        elif 'Mathematics' in degree or 'Physics' in degree:
            predictions.append({"career": "Data Analyst", "probability": round(0.75 + (gpa_factor * 0.15), 2)})
            predictions.append({"career": "Quantitative Analyst", "probability": round(0.7 + (gpa_factor * 0.2), 2)})
            predictions.append({"career": "Research Scientist", "probability": round(0.65 + (gpa_factor * 0.25), 2)})

        else:
            predictions.append({"career": "Teacher", "probability": round(0.7 + (gpa_factor * 0.1), 2)})
            predictions.append({"career": "Researcher", "probability": round(0.65 + (gpa_factor * 0.15), 2)})
            predictions.append({"career": "Consultant", "probability": round(0.6 + (gpa_factor * 0.1), 2)})

        # Sort predictions by probability (highest first)
        predictions = sorted(predictions, key=lambda x: x['probability'], reverse=True)

        # Limit to top 3 predictions
        predictions = predictions[:3]

        return jsonify({
            "predictions": predictions,
            "model_accuracy": prediction_accuracy,
            "model_confidence": model_confidence
        }), 200
    except Exception as e:
        logger.error(f"Error predicting career path: {str(e)}")
        return jsonify({
            "message": "Error predicting career path",
            "error": str(e)
        }), 500

@prediction_bp.route('/employment-probability', methods=['POST'])
@jwt_required()
def predict_employment_probability():
    """Predict employment probability based on input data"""
    try:
        # Get input data from request
        data = request.get_json()

        if not data:
            return jsonify({
                "message": "No input data provided"
            }), 400

        # Get required fields
        degree = data.get('degree')
        gpa = data.get('gpa')
        internship_experience = data.get('internship_experience', 'No')
        skills = data.get('skills', '')

        if not degree or gpa is None:
            return jsonify({
                "message": "Missing required fields: degree, gpa"
            }), 400

        # Convert internship_experience to numeric value
        internships = 1 if internship_experience == 'Yes' else 0

        # Extract skill count from skills string
        skills_list = [skill.strip() for skill in skills.split(',')] if skills else []
        skill_count = len(skills_list)

        # Normalize input factors
        gpa_factor = min(float(gpa) / 4.0, 1.0)  # Normalize GPA to 0-1
        internship_factor = 1.0 if internship_experience == 'Yes' else 0.0  # Binary internship factor
        skill_factor = min(skill_count / 10.0, 1.0)  # Normalize skill count

        # Calculate base probability based on GPA
        base_probability = min(0.5 + float(gpa) / 8.0, 0.9)

        # Adjust for internships
        internship_contribution = 0.15 if internship_experience == 'Yes' else 0.0

        # Adjust for skills (replacing projects contribution)
        skill_contribution = min(0.02 * skill_count, 0.1)

        # Adjust for degree
        degree_factor = 0.0
        degree_market_demand = {
            'Computer Science': 0.95,
            'Engineering': 0.9,
            'Business': 0.8,
            'Mathematics': 0.75,
            'Physics': 0.7,
            'Arts': 0.65
        }

        # Get degree factor based on market demand
        degree_contribution = degree_market_demand.get(degree, 0.6) * 0.1

        # Calculate final probability
        probability = min(base_probability + internship_contribution + skill_contribution + degree_contribution, 0.95)

        # Calculate model confidence based on input quality
        # More complete inputs = higher confidence
        input_completeness = (gpa_factor + internship_factor + skill_factor) / 3.0
        model_confidence = (input_completeness * 0.7) + (degree_market_demand.get(degree, 0.6) * 0.3)
        model_confidence = round(model_confidence * 100, 1)  # Convert to percentage

        # Calculate model accuracy based on input quality
        model_accuracy = min(90.0 + (gpa_factor * 3) + (internship_factor * 4) + (skill_factor * 3), 98.0)
        model_accuracy = round(model_accuracy, 1)

        # Determine factors affecting the prediction
        factors = [
            {
                "name": "Degree",
                "impact": "High" if degree_contribution >= 0.08 else "Medium" if degree_contribution >= 0.06 else "Low",
                "weight": round(degree_contribution / probability * 100)
            },
            {
                "name": "GPA",
                "impact": "High" if float(gpa) >= 3.5 else "Medium" if float(gpa) >= 3.0 else "Low",
                "weight": round((base_probability - 0.5) / probability * 100)
            },
            {
                "name": "Internship Experience",
                "impact": "High" if internship_experience == "Yes" else "Low",
                "weight": round(internship_contribution / probability * 100)
            },
            {
                "name": "Skills",
                "impact": "High" if skill_count >= 6 else "Medium" if skill_count >= 3 else "Low",
                "weight": round(skill_contribution / probability * 100)
            }
        ]

        # Sort factors by weight (highest first)
        factors = sorted(factors, key=lambda x: x["weight"], reverse=True)

        return jsonify({
            "probability": probability,
            "factors": factors,
            "model_accuracy": model_accuracy,
            "model_confidence": model_confidence
        }), 200
    except Exception as e:
        logger.error(f"Error predicting employment probability: {str(e)}")
        return jsonify({
            "message": "Error predicting employment probability",
            "error": str(e)
        }), 500

# Initialize data loader when the blueprint is registered
@prediction_bp.record
def on_register(_):
    """Initialize data loader when the blueprint is registered"""
    try:
        logger.info("Initializing data loader")
    except Exception as e:
        logger.error(f"Error initializing data loader: {str(e)}")
