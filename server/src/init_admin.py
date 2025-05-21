from pymongo import MongoClient
from werkzeug.security import generate_password_hash
from dotenv import load_dotenv
import os
import sys

def create_admin_user():
    """Create an initial admin user in the database"""
    # Load environment variables
    print("Loading environment variables...")
    load_dotenv()

    # Connect to MongoDB
    mongo_uri = os.getenv('MONGODB_URI')
    db_name = os.getenv('DATABASE_NAME')

    print(f"MongoDB URI: {mongo_uri}")
    print(f"Database Name: {db_name}")

    if not mongo_uri or not db_name:
        print("Error: MongoDB connection details not found in environment variables")
        return False

    try:
        # Connect to MongoDB
        print("Connecting to MongoDB...")
        client = MongoClient(mongo_uri)
        db = client[db_name]

        # Ping the database to verify connection
        print("Verifying connection...")
        client.admin.command('ping')
        print("Connected to MongoDB successfully!")

        # Check if admin collection exists and has users
        admin_count = db.admins.count_documents({})
        print(f"Found {admin_count} existing admin users")

        if admin_count > 0:
            print("Admin user already exists")
            return True

        # Create admin user
        print("Creating new admin user...")
        admin = {
            "name": "Admin User",
            "email": "admin@example.com",
            "password": generate_password_hash("admin123")
        }

        result = db.admins.insert_one(admin)

        if result.inserted_id:
            print(f"Admin user created successfully with ID: {result.inserted_id}")
            print("Email: admin@example.com")
            print("Password: admin123")
            return True
        else:
            print("Failed to create admin user")
            return False

    except Exception as e:
        print(f"Error creating admin user: {str(e)}")
        print(f"Exception type: {type(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Starting admin user initialization...")
    success = create_admin_user()
    print(f"Initialization {'successful' if success else 'failed'}")
    sys.exit(0 if success else 1)
