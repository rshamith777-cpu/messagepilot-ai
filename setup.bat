@echo off
echo === Step 1: Initialize Git ===
git init "C:\Users\SUMITH R\Desktop\Hackathon"
if errorlevel 1 (
    echo Git init failed. Make sure Git is installed.
    pause
    exit /b 1
)

echo.
echo === Step 2: Extract ZIP into Hackathon folder ===
powershell -NoProfile -Command ^
  "Expand-Archive -Path 'C:\Users\SUMITH R\Downloads\hackerrank-orchestrate-august26-main.zip' -DestinationPath 'C:\Users\SUMITH R\Desktop\Hackathon\__extracted__' -Force"
if errorlevel 1 (
    echo ZIP extraction failed. Check the ZIP path.
    pause
    exit /b 1
)

echo.
echo === Step 3: Move files up (flatten nested folder if needed) ===
powershell -NoProfile -Command ^
  "$src = Get-ChildItem 'C:\Users\SUMITH R\Desktop\Hackathon\__extracted__' -Directory | Select-Object -First 1; " ^
  "if ($src) { " ^
  "  Get-ChildItem $src.FullName | Move-Item -Destination 'C:\Users\SUMITH R\Desktop\Hackathon\' -Force; " ^
  "  Remove-Item $src.FullName -Recurse -Force; " ^
  "} else { " ^
  "  Get-ChildItem 'C:\Users\SUMITH R\Desktop\Hackathon\__extracted__' | Move-Item -Destination 'C:\Users\SUMITH R\Desktop\Hackathon\' -Force; " ^
  "} " ^
  "Remove-Item 'C:\Users\SUMITH R\Desktop\Hackathon\__extracted__' -Recurse -Force -ErrorAction SilentlyContinue"

echo.
echo === Step 4: Create hackathon-build branch ===
cd /d "C:\Users\SUMITH R\Desktop\Hackathon"
git add -A
git commit -m "Initial commit: HackerRank starter files"
git checkout -b hackathon-build

echo.
echo === DONE! Repository is ready. ===
echo Contents:
dir /b "C:\Users\SUMITH R\Desktop\Hackathon"
pause
