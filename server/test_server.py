from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/api/admin/login', methods=['POST'])
def login():
    """Test login endpoint"""
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')
    
    print(f"Login attempt: {email}, {password}")
    
    if email == 'admin@alumni.edu' and password == 'admin123':
        return jsonify({
            "message": "Login successful",
            "token": "test_token",
            "user": {
                "id": "1",
                "name": "Admin User",
                "email": email
            }
        }), 200
    else:
        return jsonify({
            "message": "Invalid email or password"
        }), 401

@app.route('/api/cors-test', methods=['GET'])
def cors_test():
    """Test CORS configuration"""
    return jsonify({
        "message": "CORS is working correctly",
        "origin": request.headers.get('Origin', 'Unknown')
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
