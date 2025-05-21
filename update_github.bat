@echo off
echo Updating Alumni Management System Server on GitHub

REM Add all files to Git
echo Adding files to Git...
git add .

REM Commit changes
echo Committing changes...
git commit -m "Update server configuration for Render deployment"

REM Push to GitHub
echo Pushing to GitHub...
git push origin main

echo Update to GitHub completed successfully!
echo Repository URL: https://github.com/mhpen/alumni-server.git
pause
