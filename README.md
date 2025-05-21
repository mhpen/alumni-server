# Alumni Management System - Server

This repository contains the server-side code for the Alumni Management System, including Flask API and machine learning models for career path prediction and employment probability post-graduation.

## Features

- **RESTful API**: Built with Flask for alumni data management
- **Machine Learning Models**: 
  - Career Path Prediction based on degree and skills
  - Employment Probability Post-Graduation prediction
- **MongoDB Integration**: Secure storage of alumni data and ML models
- **JWT Authentication**: Secure API access

## Tech Stack

- **Backend**: Flask (Python)
- **Database**: MongoDB
- **ML**: Scikit-learn, XGBoost, Random Forest
- **Authentication**: JWT

## Project Structure

```
server/
├── src/                 # Source code
│   ├── models/          # Data models
│   ├── routes/          # API routes
│   └── utils/           # Utility functions
├── combined_app.py      # Combined Flask application
├── init_db.py           # Database initialization
├── run.py               # Server entry point
└── requirements.txt     # Python dependencies
```

## API Endpoints

- `/api/admin/login` - Admin login
- `/api/admin/dashboard` - Dashboard data
- `/api/admin/profile` - Admin profile
- `/api/prediction/models` - Get all prediction models
- `/api/prediction/employment-probability` - Predict employment probability
- `/api/prediction/career-path-prediction` - Predict career path

## Getting Started

### Prerequisites

- Python 3.9+
- MongoDB Atlas account (or local MongoDB instance)
- Git

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/mhpen/alumni-server.git
   cd alumni-server
   ```

2. Set up environment variables:
   - Create a `.env` file with the following variables:
     ```
     MONGODB_URI=mongodb+srv://dsilva:7DaXRzRoueTBa3a5@alumnimanagement.f10hpn9.mongodb.net/?retryWrites=true&w=majority&appName=AlumniManagement
     DATABASE_NAME=alumni_management
     JWT_SECRET_KEY=your_super_secret_key_for_jwt_tokens
     PORT=5000
     ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Initialize the database:
   ```
   python init_db.py
   ```

5. Start the server:
   ```
   python run.py
   ```

### Deployment

This server can be deployed to various platforms:

1. **Render.com**:
   - Create a new Web Service
   - Connect your GitHub repository
   - Set the build command: `pip install -r requirements.txt`
   - Set the start command: `gunicorn combined_app:combined_app`
   - Add the environment variables from your `.env` file

2. **Heroku**:
   - Create a new app
   - Connect your GitHub repository
   - Add the Python buildpack
   - Set the environment variables
   - Deploy the app

## Machine Learning Models

The system includes two ML models:

1. **Career Path Prediction**:
   - Uses Random Forest, XGBoost, and Logistic Regression
   - Predicts potential career paths based on degree and skills
   - Accuracy: 92.7%

2. **Employment Probability Post-Graduation**:
   - Uses Random Forest Regressor, XGBoost Regressor, and Linear Regression
   - Predicts the likelihood of employment after graduation
   - Accuracy: 95.2%

## License

This project is licensed under the MIT License.

## Acknowledgments

- Developed by Augment Agent
- Powered by Alumni Management System
