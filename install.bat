@echo off
REM Script d'installation des dépendances pour Deepfake Detection Interface

echo.
echo ========================================
echo   Installation des dépendances
echo ========================================
echo.

REM Vérifie si l'environnement virtuel existe
if not exist ".venv\" (
    echo Création de l'environnement virtuel...
    python -m venv .venv
    if errorlevel 1 (
        echo ERROR: Impossible de créer l'environnement virtuel
        pause
        exit /b 1
    )
)

REM Active l'environnement virtuel
call .venv\Scripts\activate.bat

REM Upgrade pip
echo.
echo Mise à jour de pip...
python -m pip install --upgrade pip

REM Installation des dépendances
echo.
echo Installation des paquets requis...
pip install -r requirements.txt

if errorlevel 1 (
    echo.
    echo ERROR: Installation échouée!
    echo.
    pause
    exit /b 1
)

echo.
echo ========================================
echo   Installation terminée avec succès!
echo ========================================
echo.
echo Étapes suivantes:
echo 1. Placez le fichier 'best_model_fold4.pt' dans ce répertoire
echo 2. Exécutez 'run.bat' pour lancer l'interface
echo.
pause
