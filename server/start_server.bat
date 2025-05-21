@echo off
echo Starting Alumni Management System Server

REM Activate virtual environment if it exists
if exist "venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
) else (
    echo Virtual environment not found. Creating a new one...
    python -m venv venv
    call venv\Scripts\activate.bat
    
    echo Installing required packages...
    pip install -r requirements.txt
)

REM Initialize the database
echo Initializing database...
python init_db.py

REM Start the server
echo Starting server...
python run.py

pause
