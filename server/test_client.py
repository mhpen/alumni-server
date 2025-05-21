import requests
import json

def test_login():
    """Test the login endpoint"""
    url = 'http://localhost:5000/api/admin/login'
    data = {
        'email': 'admin@alumni.edu',
        'password': 'admin123'
    }
    headers = {
        'Content-Type': 'application/json'
    }
    
    try:
        response = requests.post(url, json=data, headers=headers)
        print(f"Status code: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == '__main__':
    test_login()
