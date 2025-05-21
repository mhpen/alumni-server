from flask import Blueprint, jsonify
from flask_jwt_extended import jwt_required
from bson import ObjectId
from datetime import datetime, timedelta

# Create a Blueprint for analytics routes
analytics_bp = Blueprint('analytics', __name__, url_prefix='/api/analytics')

@analytics_bp.route('/alumni-by-year', methods=['GET'])
@jwt_required()
def get_alumni_by_year():
    """Get alumni count by graduation year"""
    # Get database from app context
    from flask import current_app
    db = current_app.config['DATABASE']
    
    # Aggregate alumni by graduation year
    pipeline = [
        {"$group": {"_id": "$graduationYear", "count": {"$sum": 1}}},
        {"$sort": {"_id": 1}}
    ]
    
    result = list(db.alumni.aggregate(pipeline))
    
    # Format the response
    data = [{"year": item["_id"], "count": item["count"]} for item in result]
    
    return jsonify(data), 200

@analytics_bp.route('/alumni-by-degree', methods=['GET'])
@jwt_required()
def get_alumni_by_degree():
    """Get alumni count by degree program"""
    # Get database from app context
    from flask import current_app
    db = current_app.config['DATABASE']
    
    # Aggregate alumni by degree program
    pipeline = [
        {"$group": {"_id": "$degreeProgram", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}}
    ]
    
    result = list(db.alumni.aggregate(pipeline))
    
    # Format the response
    data = [{"degreeProgram": item["_id"], "count": item["count"]} for item in result]
    
    return jsonify(data), 200

@analytics_bp.route('/employment-status', methods=['GET'])
@jwt_required()
def get_employment_status():
    """Get alumni count by employment status"""
    # Get database from app context
    from flask import current_app
    db = current_app.config['DATABASE']
    
    # Aggregate alumni by employment status
    pipeline = [
        {"$group": {"_id": "$currentEmploymentStatus", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}}
    ]
    
    result = list(db.alumni.aggregate(pipeline))
    
    # Format the response
    data = [{"status": item["_id"], "count": item["count"]} for item in result]
    
    return jsonify(data), 200

@analytics_bp.route('/geographic-distribution', methods=['GET'])
@jwt_required()
def get_geographic_distribution():
    """Get alumni count by geographic location (local vs international)"""
    # Get database from app context
    from flask import current_app
    db = current_app.config['DATABASE']
    
    # Aggregate alumni by location
    pipeline = [
        {
            "$project": {
                "isLocal": {
                    "$cond": [
                        {"$eq": ["$location.country", "United States"]},
                        "Local",
                        "International"
                    ]
                }
            }
        },
        {"$group": {"_id": "$isLocal", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}}
    ]
    
    result = list(db.alumni.aggregate(pipeline))
    
    # Format the response
    data = [{"location": item["_id"], "count": item["count"]} for item in result]
    
    return jsonify(data), 200

@analytics_bp.route('/communication-preference', methods=['GET'])
@jwt_required()
def get_communication_preference():
    """Get alumni count by communication preference"""
    # Get database from app context
    from flask import current_app
    db = current_app.config['DATABASE']
    
    # Aggregate alumni by communication preference
    pipeline = [
        {"$group": {"_id": "$communicationPreference", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}}
    ]
    
    result = list(db.alumni.aggregate(pipeline))
    
    # Format the response
    data = [{"preference": item["_id"], "count": item["count"]} for item in result]
    
    return jsonify(data), 200

@analytics_bp.route('/alumni-engagement', methods=['GET'])
@jwt_required()
def get_alumni_engagement():
    """Get alumni engagement metrics"""
    # Get database from app context
    from flask import current_app
    db = current_app.config['DATABASE']
    
    # Get event participation count
    event_count = db.eventParticipation.count_documents({})
    
    # Get feedback count
    feedback_count = db.feedback.count_documents({})
    
    # Get login count in the last 30 days
    thirty_days_ago = datetime.now() - timedelta(days=30)
    login_count = db.loginLogs.count_documents({"timestamp": {"$gte": thirty_days_ago}})
    
    # Format the response
    data = [
        {"category": "Events", "count": event_count},
        {"category": "Surveys", "count": feedback_count},
        {"category": "Platform Logins", "count": login_count}
    ]
    
    return jsonify(data), 200

@analytics_bp.route('/program-participation', methods=['GET'])
@jwt_required()
def get_program_participation():
    """Get program participation metrics"""
    # Get database from app context
    from flask import current_app
    db = current_app.config['DATABASE']
    
    # Get all programs
    programs = list(db.programs.find({}, {"_id": 1, "name": 1, "type": 1}))
    
    # Count participants for each program
    result = []
    for program in programs:
        # Count alumni who participated in this program
        count = db.eventParticipation.count_documents({"programId": program["_id"]})
        
        # Add to result
        result.append({
            "program": program["name"],
            "count": count
        })
    
    # Sort by count in descending order
    result = sorted(result, key=lambda x: x["count"], reverse=True)
    
    return jsonify(result), 200
