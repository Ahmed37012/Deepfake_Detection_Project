@echo off
REM Script pour lancer l'interface Streamlit de détection de deepfakes

echo.
echo ========================================
echo   Deepfake Detection Interface
echo ========================================
echo.

REM Vérifie que le fichier du modèle existe
if not exist "best_model_fold4.pt" (
    echo.
    echo ERROR: best_model_fold4.pt not found!
    echo.
    echo Assurez-vous que le fichier du modèle est présent dans le répertoire courant.
    echo.
    pause
    exit /b 1
)

REM Active l'environnement virtuel
call .venv\Scripts\activate.bat

REM Lance Streamlit
echo Démarrage de l'application Streamlit...
echo.
streamlit run app.py

pause
