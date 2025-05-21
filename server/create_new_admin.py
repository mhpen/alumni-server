"""
Script to create a new admin user for the Alumni Management System.
"""

import sys
import os
from werkzeug.security import generate_password_hash
from pymongo import MongoClient
from dotenv import load_dotenv

# Add the current directory to the path so we can import the src package
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Load environment variables
load_dotenv()

def create_new_admin():
    """Create a new admin user."""
    try:
        print("Starting new admin creation process...")
        
        # Connect to MongoDB
        mongo_uri = os.getenv('MONGODB_URI', 'mongodb+srv://dsilva:7DaXRzRoueTBa3a5@alumnimanagement.f10hpn9.mongodb.net/?retryWrites=true&w=majority&appName=AlumniManagement')
        db_name = os.getenv('DATABASE_NAME', 'alumni_management')
        
        print(f"Connecting to MongoDB at: {mongo_uri}")
        print(f"Using database: {db_name}")
        
        # Set a timeout for the connection
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        
        # Force a connection to verify it works
        client.server_info()
        
        print("Successfully connected to MongoDB")
        
        db = client[db_name]
        
        # Create new admin
        admin_data = {
            "email": "admin@alumni.edu",
            "password_hash": generate_password_hash("admin123"),
            "name": "Admin User",
            "role": "Administrator"
        }
        
        # Check if this admin already exists
        existing_admin = db.admins.find_one({"email": admin_data["email"]})
        
        if existing_admin:
            print(f"Admin with email {admin_data['email']} already exists. Updating password...")
            db.admins.update_one(
                {"email": admin_data["email"]},
                {"$set": {"password_hash": admin_data["password_hash"]}}
            )
            print("Admin password updated successfully!")
        else:
            # Insert admin into database
            result = db.admins.insert_one(admin_data)
            print(f"New admin user created with ID: {result.inserted_id}")
        
        print("Email: admin@alumni.edu")
        print("Password: admin123")
            
        client.close()
        print("MongoDB connection closed")
        
    except Exception as e:
        print(f"Error creating admin: {str(e)}")

if __name__ == "__main__":
    create_new_admin()
    print("Script execution completed")
