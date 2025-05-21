import os
import sys
from flask import Flask, send_from_directory, jsonify, request

# Add app directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the blueprints
from src.routes.admin_routes import admin_bp
from src.routes.analytics_routes import analytics_bp
from src.routes.prediction_routes import prediction_bp

# Create Flask app
combined_app = Flask(__name__, static_folder="../client/build")

# Configure app
from flask_cors import CORS
from flask_jwt_extended import JWTManager
from pymongo import MongoClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure CORS
CORS(combined_app)

# Configure JWT
combined_app.config["JWT_SECRET_KEY"] = os.getenv("JWT_SECRET_KEY", "your_super_secret_key_for_jwt_tokens")
jwt = JWTManager(combined_app)

# Configure MongoDB
mongodb_uri = os.getenv("MONGODB_URI", "mongodb+srv://dsilva:7DaXRzRoueTBa3a5@alumnimanagement.f10hpn9.mongodb.net/?retryWrites=true&w=majority&appName=AlumniManagement")
database_name = os.getenv("DATABASE_NAME", "alumni_management")
client = MongoClient(mongodb_uri)
db = client[database_name]
combined_app.config["DATABASE"] = db

# Register blueprints
combined_app.register_blueprint(admin_bp)
combined_app.register_blueprint(analytics_bp)
combined_app.register_blueprint(prediction_bp)

# API home route
@combined_app.route("/api")
def api_home():
    return jsonify({"message": "Alumni Management System API"})

# Health check endpoint
@combined_app.route("/api/health")
def health_check():
    return jsonify({"status": "healthy"})

# CORS test endpoint
@combined_app.route("/api/cors-test")
def cors_test():
    return jsonify({
        "message": "CORS is working correctly",
        "origin": request.headers.get("Origin", "Unknown")
    })

# Serve React app
@combined_app.route("/", defaults={"path": ""})
@combined_app.route("/<path:path>")
def serve(path):
    if path.startswith("api/"):
        # Let API routes handle this
        pass
    elif path and os.path.exists(os.path.join(combined_app.static_folder, path)):
        return send_from_directory(combined_app.static_folder, path)
    else:
        # Serve index.html for client-side routing
        return send_from_directory(combined_app.static_folder, "index.html")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    combined_app.run(host="0.0.0.0", port=port)
