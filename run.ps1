# run.ps1 - Activate venv and run the Flask app (PowerShell)
# Usage: Open PowerShell in the project root and run: .\run.ps1

Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process -Force
$venvActivate = Join-Path $PSScriptRoot 'venv\Scripts\Activate.ps1'
if (Test-Path $venvActivate) {
    & $venvActivate
} else {
    Write-Error "Virtualenv activate script not found at $venvActivate"
    exit 1
}

$env:FLASK_APP = 'app.app'
$env:FLASK_ENV = 'development'
flask run
