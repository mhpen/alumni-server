@echo off
echo Career Path Prediction Model Results Visualization

REM Activate virtual environment if it exists
if exist "server\venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call server\venv\Scripts\activate.bat
) else if exist "venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
) else (
    echo Virtual environment not found. Make sure you have run run_career_path_model.bat first.
    exit /b 1
)

REM Run the visualization script
echo Visualizing model results from MongoDB...
python visualize_model_results.py

REM Check if the script ran successfully
if %ERRORLEVEL% NEQ 0 (
    echo Visualization failed with error code %ERRORLEVEL%
    exit /b %ERRORLEVEL%
)

REM Open the visualizations directory
echo Opening visualizations directory...
start visualizations

echo Career Path Prediction model results visualization completed successfully.
pause
