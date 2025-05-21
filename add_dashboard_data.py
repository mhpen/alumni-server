"""
Script to add more realistic data to the MongoDB collections for dashboard visualization.
This script will add data to the following collections:
- alumni
- events
- eventParticipation
- feedback
- programs
"""

import pymongo
import random
import datetime
from bson import ObjectId
import logging
import json
from faker import Faker

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Faker for generating realistic data
fake = Faker()

# MongoDB connection string
MONGO_URI = "mongodb+srv://dsilva:7DaXRzRoueTBa3a5@alumnimanagement.f10hpn9.mongodb.net/?retryWrites=true&w=majority&appName=AlumniManagement"
DB_NAME = "alumni_management"

# Constants for data generation
DEGREE_PROGRAMS = [
    "Bachelor of Science in Computer Science",
    "Bachelor of Science in Information Technology",
    "Bachelor of Science in Business Administration",
    "Bachelor of Arts in Communication",
    "Bachelor of Science in Nursing",
    "Bachelor of Science in Civil Engineering",
    "Bachelor of Science in Mechanical Engineering",
    "Bachelor of Science in Electrical Engineering",
    "Bachelor of Arts in Psychology",
    "Bachelor of Science in Accounting"
]

MAJORS = [
    "Computer Science", "Information Technology", "Business Administration",
    "Communication", "Nursing", "Civil Engineering", "Mechanical Engineering",
    "Electrical Engineering", "Psychology", "Accounting", "Marketing", "Finance"
]

LOCATIONS = [
    "New York, NY", "Los Angeles, CA", "Chicago, IL", "Houston, TX", "Phoenix, AZ",
    "Philadelphia, PA", "San Antonio, TX", "San Diego, CA", "Dallas, TX", "San Jose, CA",
    "Austin, TX", "Jacksonville, FL", "Fort Worth, TX", "Columbus, OH", "Charlotte, NC"
]

SKILLS = [
    "Python", "Java", "JavaScript", "SQL", "Data Analysis", "Machine Learning",
    "Web Development", "Cloud Computing", "Project Management", "Communication",
    "Leadership", "Problem Solving", "Critical Thinking", "Teamwork", "Time Management",
    "Marketing", "Sales", "Finance", "Accounting", "Human Resources", "Customer Service",
    "Public Speaking", "Writing", "Research", "Design", "UX/UI", "Mobile Development",
    "DevOps", "Cybersecurity", "Networking", "Database Management", "Big Data"
]

EMPLOYMENT_STATUSES = [
    "Employed Full-Time", "Employed Part-Time", "Self-Employed", "Freelance",
    "Unemployed", "Pursuing Higher Education", "Internship", "Contract Work"
]

INDUSTRIES = [
    "Technology", "Healthcare", "Finance", "Education", "Manufacturing", "Retail",
    "Government", "Non-Profit", "Entertainment", "Media", "Consulting", "Energy",
    "Transportation", "Construction", "Real Estate", "Hospitality", "Agriculture"
]

COMPANY_TYPES = [
    "Startup", "Small Business", "Medium Enterprise", "Large Corporation",
    "Multinational", "Government Agency", "Non-Profit Organization", "Educational Institution"
]

SALARY_RANGES = [
    "$30,000 - $50,000", "$50,000 - $70,000", "$70,000 - $90,000",
    "$90,000 - $110,000", "$110,000 - $130,000", "$130,000+"
]

EVENT_TYPES = [
    "Career Fair", "Networking Event", "Workshop", "Seminar", "Alumni Reunion",
    "Fundraising Event", "Mentorship Program", "Panel Discussion", "Webinar",
    "Conference", "Hackathon", "Social Gathering"
]

FEEDBACK_TYPES = [
    "Career Services", "Alumni Network", "Events", "Mentorship Program",
    "Website/Portal", "Communication", "General"
]

def connect_to_mongodb():
    """Connect to MongoDB and return the database object."""
    try:
        client = pymongo.MongoClient(MONGO_URI)
        db = client[DB_NAME]
        logger.info(f"Connected to MongoDB: {DB_NAME}")
        return client, db
    except Exception as e:
        logger.error(f"Error connecting to MongoDB: {str(e)}")
        raise

def generate_alumni_data(count=200):
    """Generate realistic alumni data."""
    alumni_data = []
    current_year = datetime.datetime.now().year

    for _ in range(count):
        # Generate graduation year between 5-15 years ago
        graduation_year = random.randint(current_year - 15, current_year - 1)

        # Generate employment data
        employed_after_grad = random.choices(["Yes", "No"], weights=[0.85, 0.15])[0]

        if employed_after_grad == "Yes":
            time_to_employment = random.randint(1, 24)  # 1-24 months
        else:
            time_to_employment = None

        # Generate random skills (3-8 skills per alumni)
        num_skills = random.randint(3, 8)
        skills_list = random.sample(SKILLS, num_skills)
        skills = ", ".join(skills_list)

        # Generate degree and major
        degree = random.choice(DEGREE_PROGRAMS)
        major = degree.split("in ")[-1] if "in " in degree else random.choice(MAJORS)

        # Generate employment data
        current_employment_status = random.choice(EMPLOYMENT_STATUSES)

        if current_employment_status in ["Employed Full-Time", "Employed Part-Time", "Self-Employed", "Freelance", "Contract Work"]:
            industry = random.choice(INDUSTRIES)
            company_type = random.choice(COMPANY_TYPES)
            salary_range = random.choice(SALARY_RANGES)
            job_title = fake.job()
            skills_used = ", ".join(random.sample(skills_list, min(len(skills_list), random.randint(2, 5))))
        else:
            industry = None
            company_type = None
            salary_range = None
            job_title = None
            skills_used = None

        alumni = {
            "name": fake.name(),
            "age": random.randint(22, 45),
            "gender": random.choice(["Male", "Female", "Non-binary", "Prefer not to say"]),
            "graduation_year": graduation_year,
            "degree": degree,
            "major": major,
            "gpa": round(random.uniform(2.5, 4.0), 2),
            "internship_experience": random.choice(["Yes", "No"]),
            "skills": skills,
            "location": random.choice(LOCATIONS),
            "employed_after_grad": employed_after_grad,
            "time_to_employment": time_to_employment,
            "current_employment_status": current_employment_status,
            "industry": industry,
            "company_type": company_type,
            "salary_range": salary_range,
            "job_title": job_title,
            "skills_used": skills_used,
            "email": fake.email(),
            "phone": fake.phone_number(),
            "linkedin_profile": f"linkedin.com/in/{fake.user_name()}",
            "communication_preference": random.choice(["Email", "Phone", "Both"]),
            "created_at": datetime.datetime.now(),
            "updated_at": datetime.datetime.now()
        }

        alumni_data.append(alumni)

    return alumni_data

def generate_events_data(count=30):
    """Generate realistic events data."""
    events_data = []
    current_year = datetime.datetime.now().year

    for _ in range(count):
        # Generate event date (past and future events)
        event_date = fake.date_between(start_date=f"-{random.randint(1, 3)}y", end_date=f"+{random.randint(0, 1)}y")
        # Convert date to datetime for MongoDB compatibility
        event_date = datetime.datetime.combine(event_date, datetime.time())

        event = {
            "event_name": f"{random.choice(EVENT_TYPES)} - {fake.bs()}",
            "event_type": random.choice(EVENT_TYPES),
            "description": fake.paragraph(nb_sentences=3),
            "date": event_date,
            "location": random.choice([fake.address(), "Virtual"]),
            "organizer": fake.company(),
            "max_participants": random.randint(20, 200),
            "registration_required": random.choice([True, False]),
            "created_at": datetime.datetime.now(),
            "updated_at": datetime.datetime.now()
        }

        events_data.append(event)

    return events_data

def generate_event_participation_data(alumni_ids, event_ids, count=150):
    """Generate realistic event participation data."""
    participation_data = []

    for _ in range(count):
        alumni_id = random.choice(alumni_ids)
        event_id = random.choice(event_ids)

        participation = {
            "alumni_id": alumni_id,
            "event_id": event_id,
            "registration_date": fake.date_time_between(start_date="-1y", end_date="now"),
            "attendance_status": random.choice(["Registered", "Attended", "Cancelled", "No-show"]),
            "feedback_provided": random.choice([True, False]),
            "created_at": datetime.datetime.now(),
            "updated_at": datetime.datetime.now()
        }

        participation_data.append(participation)

    return participation_data

def generate_feedback_data(alumni_ids, count=100):
    """Generate realistic feedback data."""
    feedback_data = []

    for _ in range(count):
        alumni_id = random.choice(alumni_ids)

        feedback = {
            "alumni_id": alumni_id,
            "service_type": random.choice(FEEDBACK_TYPES),
            "rating": random.randint(1, 5),
            "comments": fake.paragraph(nb_sentences=2),
            "suggestions": fake.paragraph(nb_sentences=1) if random.choice([True, False]) else "",
            "created_at": fake.date_time_between(start_date="-1y", end_date="now"),
            "updated_at": datetime.datetime.now()
        }

        feedback_data.append(feedback)

    return feedback_data

def generate_programs_data(count=15):
    """Generate realistic programs data."""
    programs_data = []

    program_types = [
        "Mentorship", "Career Development", "Networking", "Professional Development",
        "Leadership", "Entrepreneurship", "Community Service", "Research", "Internship"
    ]

    for _ in range(count):
        program_type = random.choice(program_types)

        program = {
            "program_name": f"{program_type} Program - {fake.bs()}",
            "program_type": program_type,
            "description": fake.paragraph(nb_sentences=3),
            "start_date": datetime.datetime.combine(fake.date_between(start_date="-1y", end_date="+1m"), datetime.time()),
            "end_date": datetime.datetime.combine(fake.date_between(start_date="+1m", end_date="+1y"), datetime.time()),
            "eligibility_criteria": fake.paragraph(nb_sentences=1),
            "max_participants": random.randint(10, 100),
            "current_participants": random.randint(0, 10),
            "status": random.choice(["Active", "Upcoming", "Completed", "Cancelled"]),
            "created_at": datetime.datetime.now(),
            "updated_at": datetime.datetime.now()
        }

        programs_data.append(program)

    return programs_data

def main():
    """Main function to add data to MongoDB collections."""
    try:
        # Connect to MongoDB
        client, db = connect_to_mongodb()

        # Check existing data counts
        alumni_count = db.alumni.count_documents({})
        events_count = db.events.count_documents({})
        participation_count = db.eventParticipation.count_documents({})
        feedback_count = db.feedback.count_documents({})
        programs_count = db.programs.count_documents({})

        logger.info(f"Current data counts:")
        logger.info(f"  Alumni: {alumni_count}")
        logger.info(f"  Events: {events_count}")
        logger.info(f"  Event Participation: {participation_count}")
        logger.info(f"  Feedback: {feedback_count}")
        logger.info(f"  Programs: {programs_count}")

        # Generate and insert alumni data
        alumni_data = generate_alumni_data(count=200)
        if alumni_data:
            result = db.alumni.insert_many(alumni_data)
            logger.info(f"Added {len(result.inserted_ids)} alumni records")
            alumni_ids = [str(id) for id in result.inserted_ids]
        else:
            alumni_ids = [str(doc["_id"]) for doc in db.alumni.find({}, {"_id": 1})]

        # Generate and insert events data
        events_data = generate_events_data(count=30)
        if events_data:
            result = db.events.insert_many(events_data)
            logger.info(f"Added {len(result.inserted_ids)} event records")
            event_ids = [str(id) for id in result.inserted_ids]
        else:
            event_ids = [str(doc["_id"]) for doc in db.events.find({}, {"_id": 1})]

        # Generate and insert event participation data
        participation_data = generate_event_participation_data(alumni_ids, event_ids, count=150)
        if participation_data:
            result = db.eventParticipation.insert_many(participation_data)
            logger.info(f"Added {len(result.inserted_ids)} event participation records")

        # Generate and insert feedback data
        feedback_data = generate_feedback_data(alumni_ids, count=100)
        if feedback_data:
            result = db.feedback.insert_many(feedback_data)
            logger.info(f"Added {len(result.inserted_ids)} feedback records")

        # Generate and insert programs data
        programs_data = generate_programs_data(count=15)
        if programs_data:
            result = db.programs.insert_many(programs_data)
            logger.info(f"Added {len(result.inserted_ids)} program records")

        # Close MongoDB connection
        client.close()
        logger.info("MongoDB connection closed")
        logger.info("Data generation completed successfully")

    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")

if __name__ == "__main__":
    main()
