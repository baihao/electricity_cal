# PowerShell script to set up conda environment for electricity_cal project
# Exit on error
$ErrorActionPreference = "Stop"

$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$ENV_FILE = Join-Path $SCRIPT_DIR "environment.yml"
$ENV_NAME = if ($env:CONDA_ENV_NAME) { $env:CONDA_ENV_NAME } else { "electricity_cal" }
$OLD_ENV_NAME = ".python310"

# Check if environment file exists
if (-not (Test-Path $ENV_FILE)) {
    Write-Error "Environment file not found: $ENV_FILE"
    exit 1
}

# Check if conda is installed
try {
    $null = Get-Command conda -ErrorAction Stop
} catch {
    Write-Error "Conda not detected. Please install Anaconda or Miniconda and ensure it is in your PATH."
    exit 1
}

# Get list of conda environments
$condaEnvs = conda info --envs | ForEach-Object {
    if ($_ -match '^\s*(\S+)') {
        $matches[1]
    }
}

# Remove old environment if it exists and is different from new name
if ($ENV_NAME -ne $OLD_ENV_NAME -and $OLD_ENV_NAME -in $condaEnvs) {
    Write-Host "Old environment $OLD_ENV_NAME detected, removing..."
    conda env remove -n $OLD_ENV_NAME
}

# Check if environment already exists
if ($ENV_NAME -in $condaEnvs) {
    Write-Host "Existing environment $ENV_NAME detected, updating dependencies..."
    conda env update -n $ENV_NAME -f $ENV_FILE
} else {
    Write-Host "Creating new conda environment $ENV_NAME..."
    conda env create -n $ENV_NAME -f $ENV_FILE
}

Write-Host ""
Write-Host "Environment ready. Please activate it using:"
Write-Host "  conda activate $ENV_NAME"

