@echo off
echo Employment Probability Post-Graduation Model Training with Regression Models and MongoDB Integration

REM Activate virtual environment if it exists
if exist "server\venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call server\venv\Scripts\activate.bat
) else (
    echo Virtual environment not found. Creating a new one...
    python -m venv venv
    call venv\Scripts\activate.bat
)

REM Install required packages
echo Installing required packages...
pip install -r requirements.txt

REM Test MongoDB connection
echo Testing MongoDB connection...
python -c "from mongodb_utils import MongoDBHandler; handler = MongoDBHandler(); handler.connect(); print('MongoDB connection successful'); handler.close()"

REM Check if MongoDB connection was successful
if %ERRORLEVEL% NEQ 0 (
    echo WARNING: MongoDB connection failed. The model will still train but results won't be saved to MongoDB.
    echo Press any key to continue anyway or Ctrl+C to abort...
    pause > nul
)

REM Run the model
echo Running Employment Probability Post-Graduation model with regression models...
echo This model uses:
echo  - Random Forest Regressor
echo  - XGBoost Regressor
echo  - Linear Regression (Ridge)
echo.
echo Optimized for high accuracy (target: 95%%)
echo.
python employment_probability_model.py

REM Check if the script ran successfully
if %ERRORLEVEL% NEQ 0 (
    echo Model training failed with error code %ERRORLEVEL%
    exit /b %ERRORLEVEL%
)

echo Employment Probability Post-Graduation model training completed successfully.
echo Results have been saved to MongoDB for visualization.
echo.
echo Check the employment_logs directory for detailed reports and visualizations.
pause
