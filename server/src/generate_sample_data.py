from pymongo import MongoClient
from dotenv import load_dotenv
import os
import random
from datetime import datetime, timedelta
from bson import ObjectId
import faker

# Load environment variables
load_dotenv()

# Connect to MongoDB
mongo_uri = os.getenv('MONGODB_URI')
db_name = os.getenv('DATABASE_NAME')

if not mongo_uri or not db_name:
    print("Error: MongoDB connection details not found in environment variables")
    exit(1)

client = MongoClient(mongo_uri)
db = client[db_name]

# Create Faker instance
fake = faker.Faker()

# Constants
DEGREE_PROGRAMS = [
    "Computer Science", 
    "Business Administration", 
    "Engineering", 
    "Arts & Humanities", 
    "Medicine", 
    "Law"
]

EMPLOYMENT_STATUSES = [
    "Employed", 
    "Self-employed", 
    "Unemployed", 
    "Further Studies"
]

COMMUNICATION_PREFERENCES = [
    "Email", 
    "SMS", 
    "Social Media"
]

EVENT_TYPES = [
    "Webinar", 
    "Meetup", 
    "Reunion", 
    "Workshop", 
    "Career Fair"
]

PROGRAM_TYPES = [
    "Mentorship", 
    "Networking Events", 
    "Career Fairs", 
    "Workshops", 
    "Alumni Reunions"
]

SKILLS = [
    "Programming", 
    "Data Analysis", 
    "Project Management", 
    "Digital Marketing", 
    "Leadership", 
    "Communication", 
    "Problem Solving", 
    "Critical Thinking", 
    "Teamwork", 
    "Creativity"
]

INDUSTRIES = [
    "Technology", 
    "Finance", 
    "Healthcare", 
    "Education", 
    "Manufacturing", 
    "Retail", 
    "Media", 
    "Government"
]

SERVICE_TYPES = [
    "Career Counseling", 
    "Resume Review", 
    "Interview Preparation", 
    "Job Placement", 
    "Networking"
]

# Generate Alumni Data
def generate_alumni(count=40):
    alumni_ids = []
    
    for _ in range(count):
        graduation_year = random.randint(2018, 2023)
        
        alumni = {
            "fullName": fake.name(),
            "email": fake.email(),
            "phone": fake.phone_number(),
            "graduationYear": graduation_year,
            "degreeProgram": random.choice(DEGREE_PROGRAMS),
            "location": {
                "country": "United States" if random.random() < 0.7 else fake.country(),
                "city": fake.city()
            },
            "currentEmploymentStatus": random.choice(EMPLOYMENT_STATUSES),
            "communicationPreference": random.choice(COMMUNICATION_PREFERENCES),
            "registeredAt": fake.date_time_between(start_date="-2y", end_date="now"),
            "lastLogin": fake.date_time_between(start_date="-1m", end_date="now")
        }
        
        result = db.alumni.insert_one(alumni)
        alumni_ids.append(result.inserted_id)
        
    return alumni_ids

# Generate Employment Records
def generate_employment_records(alumni_ids, count=40):
    for _ in range(count):
        alumni_id = random.choice(alumni_ids)
        
        start_date = fake.date_time_between(start_date="-5y", end_date="-1y")
        is_current = random.random() < 0.7
        
        employment_record = {
            "alumniId": alumni_id,
            "companyName": fake.company(),
            "position": fake.job(),
            "industry": random.choice(INDUSTRIES),
            "startDate": start_date,
            "endDate": None if is_current else fake.date_time_between(start_date=start_date, end_date="now"),
            "isCurrent": is_current
        }
        
        db.employmentRecords.insert_one(employment_record)

# Generate Events
def generate_events(count=40):
    event_ids = []
    
    for _ in range(count):
        event = {
            "title": fake.sentence(nb_words=4),
            "date": fake.date_time_between(start_date="-1y", end_date="+6m"),
            "type": random.choice(EVENT_TYPES),
            "location": fake.address()
        }
        
        result = db.events.insert_one(event)
        event_ids.append(result.inserted_id)
        
    return event_ids

# Generate Event Participation
def generate_event_participation(alumni_ids, event_ids, count=40):
    for _ in range(count):
        alumni_id = random.choice(alumni_ids)
        event_id = random.choice(event_ids)
        
        event_participation = {
            "alumniId": alumni_id,
            "eventId": event_id,
            "status": random.choice(["Registered", "Attended", "Cancelled"]),
            "feedbackGiven": random.random() < 0.5
        }
        
        db.eventParticipation.insert_one(event_participation)

# Generate Programs
def generate_programs(count=40):
    program_ids = []
    
    for _ in range(count):
        program = {
            "name": fake.sentence(nb_words=3),
            "description": fake.paragraph(),
            "type": random.choice(PROGRAM_TYPES),
            "createdAt": fake.date_time_between(start_date="-2y", end_date="now")
        }
        
        result = db.programs.insert_one(program)
        program_ids.append(result.inserted_id)
        
    return program_ids

# Generate Skills
def generate_skills(alumni_ids, count=40):
    for _ in range(count):
        alumni_id = random.choice(alumni_ids)
        
        skill = {
            "alumniId": alumni_id,
            "skillName": random.choice(SKILLS),
            "certificationName": fake.sentence(nb_words=3) if random.random() < 0.5 else None,
            "issuedBy": fake.company() if random.random() < 0.5 else None,
            "dateEarned": fake.date_time_between(start_date="-3y", end_date="now")
        }
        
        db.skills.insert_one(skill)

# Generate Feedback
def generate_feedback(alumni_ids, count=40):
    for _ in range(count):
        alumni_id = random.choice(alumni_ids)
        
        feedback = {
            "alumniId": alumni_id,
            "serviceType": random.choice(SERVICE_TYPES),
            "rating": random.randint(1, 5),
            "comments": fake.paragraph(),
            "createdAt": fake.date_time_between(start_date="-1y", end_date="now")
        }
        
        db.feedback.insert_one(feedback)

# Generate Login Logs
def generate_login_logs(alumni_ids, count=40):
    for _ in range(count):
        alumni_id = random.choice(alumni_ids)
        
        login_log = {
            "alumniId": alumni_id,
            "timestamp": fake.date_time_between(start_date="-30d", end_date="now"),
            "device": random.choice(["Desktop", "Mobile", "Tablet"]),
            "ipAddress": fake.ipv4()
        }
        
        db.loginLogs.insert_one(login_log)

# Main function to generate all data
def generate_all_data():
    # Clear existing data
    db.alumni.delete_many({})
    db.employmentRecords.delete_many({})
    db.events.delete_many({})
    db.eventParticipation.delete_many({})
    db.programs.delete_many({})
    db.skills.delete_many({})
    db.feedback.delete_many({})
    db.loginLogs.delete_many({})
    
    print("Generating alumni data...")
    alumni_ids = generate_alumni(40)
    
    print("Generating employment records...")
    generate_employment_records(alumni_ids, 40)
    
    print("Generating events...")
    event_ids = generate_events(40)
    
    print("Generating event participation...")
    generate_event_participation(alumni_ids, event_ids, 40)
    
    print("Generating programs...")
    program_ids = generate_programs(40)
    
    print("Generating skills...")
    generate_skills(alumni_ids, 40)
    
    print("Generating feedback...")
    generate_feedback(alumni_ids, 40)
    
    print("Generating login logs...")
    generate_login_logs(alumni_ids, 40)
    
    print("Sample data generation complete!")

if __name__ == "__main__":
    generate_all_data()
