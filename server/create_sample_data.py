import sys
import os
from pymongo import MongoClient
from faker import Faker
from datetime import datetime, timedelta
import random
from bson import ObjectId
from dotenv import load_dotenv

# Add the current directory to the path so we can import the src package
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Load environment variables
load_dotenv()

# Initialize Faker
fake = Faker()

def create_sample_data():
    """Create sample data for the Alumni Management System"""
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
        
        # Create sample alumni
        create_alumni(db, 50)
        
        # Create sample programs
        program_ids = create_programs(db, 10)
        
        # Create sample event participation
        create_event_participation(db, program_ids)
        
        # Create sample feedback
        create_feedback(db)
        
        # Create sample login logs
        create_login_logs(db)
        
        print("Sample data created successfully!")
        return True
            
    except Exception as e:
        print(f"Error creating sample data: {str(e)}")
        return False

def create_alumni(db, count=50):
    """Create sample alumni records"""
    # Check if alumni already exist
    existing_count = db.alumni.count_documents({})
    if existing_count > 0:
        print(f"Alumni collection already has {existing_count} records. Skipping creation.")
        return
    
    print(f"Creating {count} sample alumni records...")
    
    degree_programs = [
        "Computer Science", "Business Administration", "Engineering", 
        "Arts", "Science", "Medicine", "Law", "Education"
    ]
    
    employment_statuses = [
        "Employed", "Unemployed", "Self-employed", "Student", "Retired"
    ]
    
    communication_preferences = [
        "Email", "Phone", "Mail", "SMS", "No Contact"
    ]
    
    alumni_records = []
    
    for _ in range(count):
        graduation_year = random.randint(2000, 2023)
        
        alumni = {
            "firstName": fake.first_name(),
            "lastName": fake.last_name(),
            "email": fake.email(),
            "graduationYear": graduation_year,
            "degreeProgram": random.choice(degree_programs),
            "currentEmploymentStatus": random.choice(employment_statuses),
            "company": fake.company() if random.random() > 0.3 else None,
            "jobTitle": fake.job() if random.random() > 0.3 else None,
            "location": {
                "city": fake.city(),
                "state": fake.state(),
                "country": "United States" if random.random() > 0.3 else fake.country()
            },
            "communicationPreference": random.choice(communication_preferences),
            "createdAt": fake.date_time_between(start_date="-3y", end_date="now")
        }
        
        alumni_records.append(alumni)
    
    if alumni_records:
        db.alumni.insert_many(alumni_records)
        print(f"Created {len(alumni_records)} alumni records.")

def create_programs(db, count=10):
    """Create sample programs and return their IDs"""
    # Check if programs already exist
    existing_count = db.programs.count_documents({})
    if existing_count > 0:
        print(f"Programs collection already has {existing_count} records. Skipping creation.")
        program_ids = [doc["_id"] for doc in db.programs.find({}, {"_id": 1})]
        return program_ids
    
    print(f"Creating {count} sample programs...")
    
    program_types = ["Workshop", "Seminar", "Networking", "Career Fair", "Alumni Meetup"]
    
    program_records = []
    
    for _ in range(count):
        program = {
            "name": fake.catch_phrase(),
            "description": fake.paragraph(),
            "type": random.choice(program_types),
            "date": fake.date_time_between(start_date="-1y", end_date="+6m"),
            "location": fake.address(),
            "createdAt": fake.date_time_between(start_date="-2y", end_date="now")
        }
        
        program_records.append(program)
    
    if program_records:
        result = db.programs.insert_many(program_records)
        print(f"Created {len(program_records)} program records.")
        return result.inserted_ids
    
    return []

def create_event_participation(db, program_ids):
    """Create sample event participation records"""
    # Check if event participation records already exist
    existing_count = db.eventParticipation.count_documents({})
    if existing_count > 0:
        print(f"Event participation collection already has {existing_count} records. Skipping creation.")
        return
    
    if not program_ids:
        print("No program IDs available. Skipping event participation creation.")
        return
    
    print("Creating sample event participation records...")
    
    # Get all alumni IDs
    alumni_ids = [doc["_id"] for doc in db.alumni.find({}, {"_id": 1})]
    
    if not alumni_ids:
        print("No alumni found. Skipping event participation creation.")
        return
    
    participation_records = []
    
    # Create random participation records
    for program_id in program_ids:
        # Randomly select 5-20 alumni for each program
        participant_count = random.randint(5, min(20, len(alumni_ids)))
        participants = random.sample(alumni_ids, participant_count)
        
        for alumni_id in participants:
            participation = {
                "alumniId": alumni_id,
                "programId": program_id,
                "registrationDate": fake.date_time_between(start_date="-6m", end_date="now"),
                "attended": random.random() > 0.2,  # 80% chance of attending
                "feedback": fake.paragraph() if random.random() > 0.7 else None
            }
            
            participation_records.append(participation)
    
    if participation_records:
        db.eventParticipation.insert_many(participation_records)
        print(f"Created {len(participation_records)} event participation records.")

def create_feedback(db):
    """Create sample feedback records"""
    # Check if feedback records already exist
    existing_count = db.feedback.count_documents({})
    if existing_count > 0:
        print(f"Feedback collection already has {existing_count} records. Skipping creation.")
        return
    
    print("Creating sample feedback records...")
    
    # Get all alumni IDs
    alumni_ids = [doc["_id"] for doc in db.alumni.find({}, {"_id": 1})]
    
    if not alumni_ids:
        print("No alumni found. Skipping feedback creation.")
        return
    
    feedback_records = []
    
    # Create 20-30 feedback records
    feedback_count = random.randint(20, 30)
    
    for _ in range(feedback_count):
        feedback = {
            "alumniId": random.choice(alumni_ids),
            "topic": fake.bs(),
            "content": fake.paragraph(),
            "rating": random.randint(1, 5),
            "submittedAt": fake.date_time_between(start_date="-1y", end_date="now")
        }
        
        feedback_records.append(feedback)
    
    if feedback_records:
        db.feedback.insert_many(feedback_records)
        print(f"Created {len(feedback_records)} feedback records.")

def create_login_logs(db):
    """Create sample login logs"""
    # Check if login logs already exist
    existing_count = db.loginLogs.count_documents({})
    if existing_count > 0:
        print(f"Login logs collection already has {existing_count} records. Skipping creation.")
        return
    
    print("Creating sample login logs...")
    
    # Get all alumni IDs
    alumni_ids = [doc["_id"] for doc in db.alumni.find({}, {"_id": 1})]
    
    if not alumni_ids:
        print("No alumni found. Skipping login logs creation.")
        return
    
    login_records = []
    
    # Create 100-200 login records
    login_count = random.randint(100, 200)
    
    for _ in range(login_count):
        # Random alumni
        alumni_id = random.choice(alumni_ids)
        
        # Random login time in the last 30 days
        login_time = fake.date_time_between(start_date="-30d", end_date="now")
        
        login = {
            "userId": alumni_id,
            "userType": "alumni",
            "timestamp": login_time,
            "ipAddress": fake.ipv4(),
            "userAgent": fake.user_agent()
        }
        
        login_records.append(login)
    
    if login_records:
        db.loginLogs.insert_many(login_records)
        print(f"Created {len(login_records)} login log records.")

if __name__ == "__main__":
    create_sample_data()
