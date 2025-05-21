@echo off
echo Updating Alumni Management System Frontend

REM Add all files to Git
echo Adding files to Git...
git add .

REM Commit changes
echo Committing changes...
git commit -m "Update frontend to connect to deployed server"

REM Push to GitHub
echo Pushing to GitHub...
git push origin main

echo Update to GitHub completed successfully!
pause
