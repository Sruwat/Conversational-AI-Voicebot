param(
    [ValidateSet("desktop", "api")]
    [string]$Mode = "desktop",
    [string]$BindHost = "127.0.0.1",
    [int]$Port = 8000
)

$ErrorActionPreference = "Stop"

function Resolve-Python {
    $candidates = @(
        (Join-Path (Split-Path $PSScriptRoot -Parent) "venv\\Scripts\\python.exe"),
        (Join-Path (Split-Path $PSScriptRoot -Parent) ".venv\\Scripts\\python.exe"),
        (Join-Path $PSScriptRoot "venv\\Scripts\\python.exe"),
        (Join-Path $PSScriptRoot ".venv\\Scripts\\python.exe"),
        (Join-Path $PSScriptRoot "Scripts\\python.exe")
    )

    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate) {
            return $candidate
        }
    }

    $command = Get-Command python -ErrorAction SilentlyContinue
    if ($command) {
        return $command.Source
    }

    throw "No Python interpreter found. Activate a virtual environment or install Python."
}

$python = Resolve-Python
Set-Location -LiteralPath $PSScriptRoot

if ($Mode -eq "desktop") {
    Write-Host "[launch] Starting desktop voice bot with $python" -ForegroundColor Cyan
    & $python ".\\main.py"
}
else {
    Write-Host "[launch] Starting API server on http://$BindHost`:$Port with $python" -ForegroundColor Cyan
    & $python "-m" "uvicorn" "service_app:app" "--host" $BindHost "--port" $Port
}
