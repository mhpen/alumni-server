"""
Database utility functions for the Alumni Management System.
"""

from flask import current_app, g
from pymongo import MongoClient
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

def get_db():
    """
    Get the database connection.
    
    Returns:
        The MongoDB database object.
    """
    if 'db' not in g:
        # If we're in a Flask context, use the database from the app config
        if current_app:
            g.db = current_app.config['DATABASE']
        else:
            # Otherwise, create a new connection
            mongo_uri = os.getenv('MONGODB_URI')
            db_name = os.getenv('DATABASE_NAME')
            
            if not mongo_uri or not db_name:
                raise ValueError("MongoDB connection details not found in environment variables")
            
            client = MongoClient(mongo_uri)
            g.db = client[db_name]
    
    return g.db

def close_db(e=None):
    """
    Close the database connection.
    
    Args:
        e: An optional exception that occurred.
    """
    db = g.pop('db', None)
    
    if db is not None:
        # If we created our own client, close it
        if hasattr(db, 'client'):
            db.client.close()
