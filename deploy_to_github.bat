@echo off
echo Deploying Alumni Management System Server to GitHub

REM Create a new README.md for the server repository
echo Preparing README.md...
copy README-server.md README.md

REM Initialize Git repository
echo Initializing Git repository...
git init

REM Add all files to Git
echo Adding files to Git...
git add .

REM Commit changes
echo Committing changes...
git commit -m "Initial commit of Alumni Management System Server"

REM Set the main branch
echo Setting main branch...
git branch -M main

REM Add remote repository
echo Adding remote repository...
git remote add origin https://github.com/mhpen/alumni-server.git

REM Push to GitHub
echo Pushing to GitHub...
git push -u origin main

echo Deployment to GitHub completed successfully!
echo Repository URL: https://github.com/mhpen/alumni-server.git
pause
