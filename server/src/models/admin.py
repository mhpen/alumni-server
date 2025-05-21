"""
Admin model for the Alumni Management System.
"""

import datetime
import jwt
from werkzeug.security import generate_password_hash, check_password_hash
from bson import ObjectId
from src.db import get_db

class Admin:
    """Admin model for authentication and user management."""
    
    def __init__(self, email, password=None, name=None, role=None, _id=None):
        """Initialize an Admin object."""
        self.email = email
        self.password_hash = generate_password_hash(password) if password else None
        self.name = name
        self.role = role
        self._id = _id
    
    @classmethod
    def find_by_email(cls, email):
        """Find an admin by email."""
        db = get_db()
        admin_data = db.admins.find_one({"email": email})
        if admin_data:
            return cls(
                email=admin_data["email"],
                name=admin_data.get("name"),
                role=admin_data.get("role"),
                _id=admin_data["_id"]
            )
        return None
    
    @classmethod
    def find_by_id(cls, admin_id):
        """Find an admin by ID."""
        db = get_db()
        admin_data = db.admins.find_one({"_id": ObjectId(admin_id)})
        if admin_data:
            return cls(
                email=admin_data["email"],
                name=admin_data.get("name"),
                role=admin_data.get("role"),
                _id=admin_data["_id"]
            )
        return None
    
    def verify_password(self, password):
        """Verify the admin's password."""
        db = get_db()
        admin_data = db.admins.find_one({"email": self.email})
        if admin_data and "password_hash" in admin_data:
            return check_password_hash(admin_data["password_hash"], password)
        return False
    
    def save(self):
        """Save the admin to the database."""
        db = get_db()
        admin_data = {
            "email": self.email,
            "password_hash": self.password_hash,
            "name": self.name,
            "role": self.role
        }
        if self._id:
            db.admins.update_one({"_id": self._id}, {"$set": admin_data})
        else:
            result = db.admins.insert_one(admin_data)
            self._id = result.inserted_id
    
    def generate_token(self):
        """Generate a JWT token for the admin."""
        payload = {
            "admin_id": str(self._id),
            "email": self.email,
            "exp": datetime.datetime.utcnow() + datetime.timedelta(days=1)
        }
        token = jwt.encode(payload, "your-secret-key", algorithm="HS256")
        return token
    
    @staticmethod
    def verify_token(token):
        """Verify a JWT token and return the admin."""
        try:
            payload = jwt.decode(token, "your-secret-key", algorithms=["HS256"])
            admin_id = payload["admin_id"]
            return Admin.find_by_id(admin_id)
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None
    
    def to_dict(self):
        """Convert the admin to a dictionary."""
        return {
            "id": str(self._id),
            "email": self.email,
            "name": self.name,
            "role": self.role
        }
    
    @classmethod
    def create_default_admin(cls):
        """Create a default admin if none exists."""
        db = get_db()
        if db.admins.count_documents({}) == 0:
            admin = cls(
                email="admin@alumni.edu",
                password="admin123",
                name="Admin User",
                role="Administrator"
            )
            admin.save()
            return admin
        return None
