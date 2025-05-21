"""
Script to create a default admin user for the Alumni Management System.
"""

import sys
import os
import time
from werkzeug.security import generate_password_hash
from pymongo import MongoClient
from dotenv import load_dotenv

# Add the current directory to the path so we can import the src package
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Load environment variables
load_dotenv()

def create_default_admin():
    """Create a default admin user if one doesn't already exist."""
    try:
        print("Starting default admin creation process...")

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

        # Check if admin collection exists and has any users
        collection_names = db.list_collection_names()
        print(f"Available collections: {collection_names}")

        if 'admins' not in collection_names:
            print("Admin collection does not exist, creating it...")

        admin_count = db.admins.count_documents({})
        print(f"Current admin count: {admin_count}")

        if 'admins' not in collection_names or admin_count == 0:
            # Create default admin
            admin_data = {
                "email": "admin@alumni.edu",
                "password_hash": generate_password_hash("admin123"),
                "name": "Admin User",
                "role": "Administrator"
            }

            # Insert admin into database
            result = db.admins.insert_one(admin_data)
            print(f"Default admin user created with ID: {result.inserted_id}")
            print("Email: admin@alumni.edu")
            print("Password: admin123")
        else:
            print("Admin user already exists. No action taken.")

        client.close()
        print("MongoDB connection closed")

    except Exception as e:
        print(f"Error creating default admin: {str(e)}")

if __name__ == "__main__":
    create_default_admin()
    print("Script execution completed")
