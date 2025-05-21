@echo off
echo Pushing All Alumni Management System Client Files to GitHub

REM Create a temporary directory for the client code
echo Creating temporary directory...
mkdir temp_client_full

REM Copy client files to the temporary directory
echo Copying client files...
xcopy /E /I /Y client temp_client_full

REM Initialize Git in the temporary directory
echo Initializing Git repository...
cd temp_client_full
git init

REM Add all files to Git
echo Adding files to Git...
git add .

REM Commit changes
echo Committing changes...
git commit -m "Add all client files including src directory"

REM Set the main branch
echo Setting main branch...
git branch -M main

REM Add remote repository
echo Adding remote repository...
git remote add origin https://github.com/mhpen/alumni-client.git

REM Push to GitHub
echo Pushing to GitHub...
git push -u origin main --force

REM Clean up
echo Cleaning up...
cd ..
rmdir /S /Q temp_client_full

echo All client files successfully pushed to GitHub!
echo Repository URL: https://github.com/mhpen/alumni-client
pause
