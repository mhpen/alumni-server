"""
Initialize the database with default data.
"""

import sys
import os
from pymongo import MongoClient
from werkzeug.security import generate_password_hash
from dotenv import load_dotenv
from datetime import datetime

# Add the current directory to the path so we can import the src package
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Load environment variables
load_dotenv()

def init_db():
    """Initialize the database with default data."""
    # Connect to MongoDB
    client = MongoClient(os.getenv('MONGODB_URI'))
    db = client[os.getenv('DATABASE_NAME')]
    
    # Create admin user if it doesn't exist
    if db.admins.count_documents({"email": "admin@alumni.edu"}) == 0:
        db.admins.insert_one({
            "email": "admin@alumni.edu",
            "password_hash": generate_password_hash("admin123"),
            "name": "Admin User",
            "role": "Administrator",
            "created_at": datetime.now()
        })
        print("Created default admin user: admin@alumni.edu / admin123")
    else:
        print("Default admin user already exists")
    
    # Create sample alumni data if it doesn't exist
    if db.alumni.count_documents({}) == 0:
        alumni_data = [
            {
                "name": "John Smith",
                "email": "john.smith@example.com",
                "graduation_year": 2022,
                "degree": "Bachelor of Science",
                "major": "Computer Science",
                "gpa": 3.8,
                "employed_after_grad": True,
                "time_to_employment": 2,  # months
                "salary": 85000,
                "company": "Google",
                "job_title": "Software Engineer",
                "location": "Mountain View, CA",
                "skills": ["Python", "JavaScript", "Machine Learning", "Data Structures"],
                "registration_date": datetime.now()
            },
            {
                "name": "Sarah Johnson",
                "email": "sarah.johnson@example.com",
                "graduation_year": 2021,
                "degree": "Bachelor of Arts",
                "major": "Business Administration",
                "gpa": 3.6,
                "employed_after_grad": True,
                "time_to_employment": 1,  # months
                "salary": 75000,
                "company": "Amazon",
                "job_title": "Marketing Specialist",
                "location": "Seattle, WA",
                "skills": ["Marketing", "Social Media", "Content Creation", "Analytics"],
                "registration_date": datetime.now()
            },
            {
                "name": "Michael Chen",
                "email": "michael.chen@example.com",
                "graduation_year": 2022,
                "degree": "Master of Science",
                "major": "Data Science",
                "gpa": 4.0,
                "employed_after_grad": True,
                "time_to_employment": 0,  # months (had job before graduation)
                "salary": 110000,
                "company": "Microsoft",
                "job_title": "Data Scientist",
                "location": "Redmond, WA",
                "skills": ["Python", "R", "Machine Learning", "Deep Learning", "SQL"],
                "registration_date": datetime.now()
            },
            {
                "name": "Emily Davis",
                "email": "emily.davis@example.com",
                "graduation_year": 2023,
                "degree": "Bachelor of Science",
                "major": "Mechanical Engineering",
                "gpa": 3.5,
                "employed_after_grad": False,
                "skills": ["CAD", "MATLAB", "Project Management", "3D Printing"],
                "registration_date": datetime.now()
            },
            {
                "name": "David Wilson",
                "email": "david.wilson@example.com",
                "graduation_year": 2021,
                "degree": "Bachelor of Arts",
                "major": "Psychology",
                "gpa": 3.7,
                "employed_after_grad": True,
                "time_to_employment": 4,  # months
                "salary": 65000,
                "company": "City Hospital",
                "job_title": "Research Assistant",
                "location": "Boston, MA",
                "skills": ["Research", "Data Analysis", "SPSS", "Counseling"],
                "registration_date": datetime.now()
            }
        ]
        
        db.alumni.insert_many(alumni_data)
        print(f"Created {len(alumni_data)} sample alumni records")
    else:
        print("Alumni data already exists")
    
    # Create sample events
    if db.events.count_documents({}) == 0:
        events_data = [
            {
                "title": "Annual Alumni Meetup",
                "description": "Join us for our annual alumni gathering to network and reconnect with fellow graduates.",
                "date": datetime(2025, 6, 15),
                "location": "University Campus, Main Hall",
                "created_at": datetime.now()
            },
            {
                "title": "Career Fair 2025",
                "description": "Connect with top employers looking to hire our talented alumni.",
                "date": datetime(2025, 7, 10),
                "location": "University Campus, Business Building",
                "created_at": datetime.now()
            }
        ]
        
        db.events.insert_many(events_data)
        print(f"Created {len(events_data)} sample events")
    else:
        print("Events data already exists")
    
    # Create sample ML models data
    if db.ml_models.count_documents({}) == 0:
        models_data = [
            {
                "name": "Employment Probability Post-Graduation",
                "description": "Predicts the probability of employment after graduation based on student data.",
                "accuracy": 95.2,
                "created_at": datetime.now(),
                "updated_at": datetime.now()
            },
            {
                "name": "Career Path Prediction",
                "description": "Predicts potential career paths based on degree and skills.",
                "accuracy": 92.7,
                "created_at": datetime.now(),
                "updated_at": datetime.now()
            }
        ]
        
        db.ml_models.insert_many(models_data)
        print(f"Created {len(models_data)} sample ML models")
    else:
        print("ML models data already exists")
    
    print("Database initialization complete!")

if __name__ == "__main__":
    init_db()
