@echo off
REM Run the ICU Ventilator Digital Twin dashboard on Windows.
REM Opens in browser at http://localhost:8050

cd /d "%~dp0"

REM First-time dependency check
python -c "import dash, plotly, joblib, xgboost, sklearn, pandas, numpy, paho.mqtt.client" 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo Installing dependencies...
    pip install dash plotly joblib xgboost scikit-learn pandas numpy python-pptx imbalanced-learn paho-mqtt
)

echo.
echo =========================================================
echo   ICU Ventilator Digital Twin - Dashboard
echo.
echo   Open in browser:  http://localhost:8050
echo   Press Ctrl+C to stop.
echo =========================================================
echo.

REM Open the browser in the background after a short delay
start "" /B cmd /c "ping 127.0.0.1 -n 4 >nul & start http://localhost:8050"

python gui\app.py
