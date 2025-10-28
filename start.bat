@echo off
echo Starting Local RAG Chatbot...
echo.
echo Installing/updating dependencies...
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo Failed to install dependencies.
    pause
    exit /b %errorlevel%
)
echo.
echo Starting Streamlit app...
python -m streamlit run app.py
pause
