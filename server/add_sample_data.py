"""
Script to add sample data to the Alumni Management System database.
"""

import sys
import os
import random
from datetime import datetime, timedelta
from pymongo import MongoClient
from dotenv import load_dotenv

# Add the current directory to the path so we can import the src package
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Load environment variables
load_dotenv()

def add_sample_data():
    """Add sample data to the database."""
    try:
        print("Starting sample data creation process...")

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

        # Add sample alumni data
        add_alumni_data(db)

        # Add sample event participation data
        add_event_participation_data(db)

        # Add sample feedback data
        add_feedback_data(db)

        # Add sample program data
        add_program_data(db)

        # Add sample login logs
        add_login_logs(db)

        client.close()
        print("MongoDB connection closed")
        print("Sample data creation completed successfully!")

    except Exception as e:
        print(f"Error creating sample data: {str(e)}")

def add_alumni_data(db):
    """Add sample alumni data."""
    print("Adding sample alumni data...")

    # Check if alumni collection already has data
    if 'alumni' in db.list_collection_names() and db.alumni.count_documents({}) > 0:
        print(f"Alumni collection already has {db.alumni.count_documents({})} documents. Skipping...")
        return

    # Sample degree programs
    degree_programs = ['Computer Science', 'Business', 'Engineering', 'Arts', 'Mathematics', 'Physics', 'Chemistry', 'Biology', 'Economics', 'Psychology']

    # Sample employment statuses
    employment_statuses = ['Employed', 'Unemployed', 'Further Studies', 'Self-Employed', 'Internship']

    # Sample location types
    location_types = ['Local', 'International']

    # Generate 100 alumni records
    alumni_records = []
    for i in range(100):
        graduation_year = random.randint(2018, 2023)
        employed_after_grad = random.choice([True, False])

        alumni_record = {
            "name": f"Alumni {i+1}",
            "email": f"alumni{i+1}@example.com",
            "graduation_year": graduation_year,
            "degree_program": random.choice(degree_programs),
            "graduated": True,
            "employed_after_grad": employed_after_grad,
            "employment_status": random.choice(employment_statuses),
            "location_type": random.choice(location_types),
            "salary": random.randint(40000, 120000) if employed_after_grad else 0
        }

        alumni_records.append(alumni_record)

    # Insert alumni records
    if alumni_records:
        db.alumni.insert_many(alumni_records)
        print(f"Added {len(alumni_records)} alumni records")

def add_event_participation_data(db):
    """Add sample event participation data."""
    print("Adding sample event participation data...")

    # Check if eventParticipation collection already has data
    if 'eventParticipation' in db.list_collection_names() and db.eventParticipation.count_documents({}) > 0:
        print(f"Event participation collection already has {db.eventParticipation.count_documents({})} documents. Skipping...")
        return

    # Sample activity types
    activity_types = ['Events', 'Surveys', 'Platform Logins', 'Mentorship', 'Workshops', 'Webinars']

    # Generate 200 event participation records
    participation_records = []
    for i in range(200):
        participation_record = {
            "alumni_id": f"alumni_{random.randint(1, 100)}",
            "activity_type": random.choice(activity_types),
            "date": (datetime.now() - timedelta(days=random.randint(1, 365))).strftime("%Y-%m-%d"),
            "duration_minutes": random.randint(30, 240)
        }

        participation_records.append(participation_record)

    # Insert participation records
    if participation_records:
        db.eventParticipation.insert_many(participation_records)
        print(f"Added {len(participation_records)} event participation records")

def add_feedback_data(db):
    """Add sample feedback data."""
    print("Adding sample feedback data...")

    # Check if feedback collection already has data
    if 'feedback' in db.list_collection_names() and db.feedback.count_documents({}) > 0:
        print(f"Feedback collection already has {db.feedback.count_documents({})} documents. Skipping...")
        return

    # Sample service types
    service_types = ['Career Support', 'Academic Advising', 'Alumni Network', 'Events', 'Website']

    # Generate 150 feedback records
    feedback_records = []
    for i in range(150):
        feedback_record = {
            "alumni_id": f"alumni_{random.randint(1, 100)}",
            "service_type": random.choice(service_types),
            "rating": random.randint(1, 5),
            "comments": f"Sample feedback comment {i+1}",
            "date": (datetime.now() - timedelta(days=random.randint(1, 180))).strftime("%Y-%m-%d")
        }

        feedback_records.append(feedback_record)

    # Insert feedback records
    if feedback_records:
        db.feedback.insert_many(feedback_records)
        print(f"Added {len(feedback_records)} feedback records")

def add_program_data(db):
    """Add sample program data."""
    print("Adding sample program data...")

    # Check if programs collection already has data
    if 'programs' in db.list_collection_names() and db.programs.count_documents({}) > 0:
        print(f"Programs collection already has {db.programs.count_documents({})} documents. Skipping...")
        return

    # Sample programs
    programs = [
        {
            "program_name": "Annual Reunion",
            "description": "Yearly gathering of alumni",
            "date": (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d"),
            "participant_count": random.randint(80, 150)
        },
        {
            "program_name": "Career Fair",
            "description": "Connect alumni with potential employers",
            "date": (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d"),
            "participant_count": random.randint(70, 120)
        },
        {
            "program_name": "Mentorship Program",
            "description": "Alumni mentoring current students",
            "date": (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d"),
            "participant_count": random.randint(50, 100)
        },
        {
            "program_name": "Networking Event",
            "description": "Professional networking for alumni",
            "date": (datetime.now() - timedelta(days=120)).strftime("%Y-%m-%d"),
            "participant_count": random.randint(40, 80)
        },
        {
            "program_name": "Workshop Series",
            "description": "Professional development workshops",
            "date": (datetime.now() - timedelta(days=150)).strftime("%Y-%m-%d"),
            "participant_count": random.randint(30, 60)
        }
    ]

    # Insert program records
    if programs:
        db.programs.insert_many(programs)
        print(f"Added {len(programs)} program records")

def add_login_logs(db):
    """Add sample login logs."""
    print("Adding sample login logs...")

    # Check if loginLogs collection already has data
    if 'loginLogs' in db.list_collection_names() and db.loginLogs.count_documents({}) > 0:
        print(f"Login logs collection already has {db.loginLogs.count_documents({})} documents. Skipping...")
        return

    # Sample actions
    actions = ['logged in', 'updated profile', 'viewed dashboard', 'accessed models', 'logged out']

    # Sample users
    users = ['John Smith', 'Sarah Johnson', 'Michael Brown', 'Emily Davis', 'Admin User']

    # Sample timestamps
    timestamps = ['2 hours ago', '5 hours ago', 'Yesterday', '2 days ago', 'Last week']

    # Generate 20 login log records
    login_logs = []
    for i in range(20):
        login_log = {
            "user": random.choice(users),
            "action": random.choice(actions),
            "timestamp": random.choice(timestamps),
            "date": (datetime.now() - timedelta(days=random.randint(0, 7))).strftime("%Y-%m-%d")
        }

        login_logs.append(login_log)

    # Insert login log records
    if login_logs:
        db.loginLogs.insert_many(login_logs)
        print(f"Added {len(login_logs)} login log records")

if __name__ == "__main__":
    add_sample_data()
    print("Script execution completed")
