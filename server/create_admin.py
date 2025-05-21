import sys
import os
from pymongo import MongoClient
from werkzeug.security import generate_password_hash
from dotenv import load_dotenv

# Add the current directory to the path so we can import the src package
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Load environment variables
load_dotenv()

def create_admin_user():
    """Create a test admin user in the database"""
    # Connect to MongoDB
    mongo_uri = os.getenv('MONGODB_URI')
    db_name = os.getenv('DATABASE_NAME')
    
    if not mongo_uri or not db_name:
        print("Error: MongoDB connection details not found in environment variables.")
        return False
    
    try:
        # Connect to MongoDB
        client = MongoClient(mongo_uri)
        db = client[db_name]
        
        # Check if admin already exists
        existing_admin = db.admins.find_one({"email": "admin@example.com"})
        if existing_admin:
            print("Admin user already exists.")
            return True
        
        # Create admin user
        admin = {
            "name": "Admin User",
            "email": "admin@example.com",
            "password": generate_password_hash("password123")
        }
        
        result = db.admins.insert_one(admin)
        
        if result.inserted_id:
            print(f"Admin user created successfully with ID: {result.inserted_id}")
            return True
        else:
            print("Failed to create admin user.")
            return False
            
    except Exception as e:
        print(f"Error creating admin user: {str(e)}")
        return False

if __name__ == "__main__":
    create_admin_user()
