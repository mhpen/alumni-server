"""
Script to check the model-related data in the MongoDB database.
"""

import sys
import os
from pymongo import MongoClient
from dotenv import load_dotenv
import json
from bson import json_util

# Add the current directory to the path so we can import the src package
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Load environment variables
load_dotenv()

def check_model_data():
    """Check the model-related data in the MongoDB database."""
    try:
        print("Starting model data check...")
        
        # Connect to MongoDB
        mongo_uri = os.getenv('MONGODB_URI', 'mongodb+srv://dsilva:7DaXRzRoueTBa3a5@alumnimanagement.f10hpn9.mongodb.net/?retryWrites=true&w=majority&appName=AlumniManagement')
        db_name = os.getenv('DATABASE_NAME', 'alumni_management')
        
        print(f"Connecting to MongoDB at: {mongo_uri}")
        print(f"Using database: {db_name}")
        
        # Set a timeout for the connection
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=10000)
        
        # Force a connection to verify it works
        print("Checking connection...")
        client.server_info()
        
        print("Successfully connected to MongoDB")
        
        db = client[db_name]
        
        # Get list of collections
        collections = db.list_collection_names()
        print(f"Available collections: {collections}")
        
        # Check ml_models collection
        if 'ml_models' in collections:
            print("\nChecking ml_models collection...")
            models_count = db.ml_models.count_documents({})
            print(f"Total models count: {models_count}")
            
            # Get all models
            models = list(db.ml_models.find({}, {"_id": 0}))
            print("Models:")
            for model in models:
                print(f"  {json.dumps(model, default=str)}")
        else:
            print("\nml_models collection not found")
        
        # Check model_training_results collection
        if 'model_training_results' in collections:
            print("\nChecking model_training_results collection...")
            results_count = db.model_training_results.count_documents({})
            print(f"Total training results count: {results_count}")
            
            # Get all training results
            results = list(db.model_training_results.find({}, {"_id": 0}))
            print("Training results:")
            for result in results:
                print(f"  {json.dumps(result, default=str)}")
        else:
            print("\nmodel_training_results collection not found")
        
        client.close()
        print("\nMongoDB connection closed")
        
    except Exception as e:
        print(f"Error checking model data: {str(e)}")

if __name__ == "__main__":
    check_model_data()
    print("Script execution completed")
