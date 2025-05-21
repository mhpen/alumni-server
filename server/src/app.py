from flask import Flask, jsonify, request, make_response
from flask_cors import CORS
from flask_jwt_extended import JWTManager
from pymongo import MongoClient
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Configure CORS to allow requests from frontend with proper preflight handling
# Use a simpler CORS configuration that allows all origins
CORS(app, supports_credentials=True, resources={r"/*": {"origins": "*"}})

# Global OPTIONS request handler
@app.route('/', defaults={'path': ''}, methods=['OPTIONS'])
@app.route('/<path:path>', methods=['OPTIONS'])
def options_handler(path):
    response = make_response()
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    return response

# Configure JWT
app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET_KEY', 'your_super_secret_key_for_jwt_tokens')
jwt = JWTManager(app)

# Configure MongoDB
mongodb_uri = os.getenv('MONGODB_URI', 'mongodb+srv://dsilva:7DaXRzRoueTBa3a5@alumnimanagement.f10hpn9.mongodb.net/?retryWrites=true&w=majority&appName=AlumniManagement')
database_name = os.getenv('DATABASE_NAME', 'alumni_management')

# Add SSL configuration and connection timeout
client = MongoClient(
    mongodb_uri,
    ssl=True,
    ssl_cert_reqs='CERT_NONE',  # Disable certificate verification
    connectTimeoutMS=30000,
    socketTimeoutMS=30000,
    serverSelectionTimeoutMS=30000
)
db = client[database_name]
app.config['DATABASE'] = db

# Import routes
from .routes.admin_routes import admin_bp
from .routes.analytics_routes import analytics_bp
from .routes.prediction_routes import prediction_bp

# Register blueprints - FIXED: removed duplicate '/api' prefix
app.register_blueprint(admin_bp)
app.register_blueprint(analytics_bp)
app.register_blueprint(prediction_bp)

# API home route
@app.route('/api')
def api_home():
    return jsonify({'message': 'Alumni Management System API'})

# Health check endpoint
@app.route('/api/health')
def health_check():
    return jsonify({'status': 'healthy'})

# CORS test endpoint
@app.route('/api/cors-test')
def cors_test():
    return jsonify({
        'message': 'CORS is working correctly',
        'origin': request.headers.get('Origin', 'Unknown'),
        'allowed_origins': '*'
    })

# Root route for testing
@app.route('/')
def root():
    return jsonify({
        'message': 'Alumni Management System API is running',
        'status': 'online',
        'frontend_url': 'http://localhost:3000',
        'cors_enabled': True,
        'allowed_origins': '*'
    })

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'message': 'API resource not found'}), 404

@app.errorhandler(500)
def server_error(error):
    return jsonify({'message': 'Internal server error'}), 500

if __name__ == '__main__':
    # Get port from environment variable or default to 8080
    port = int(os.environ.get('PORT', 8080))
    print(f"Starting server on port {port}")
    app.run(host='0.0.0.0', port=port)
