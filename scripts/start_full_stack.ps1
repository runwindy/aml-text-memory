param(
    [Alias("Host")]
    [string]$BindHost = "127.0.0.1",
    [int]$Port = 8000,
    [switch]$Reload
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$envFile = Join-Path $projectRoot ".env"

if (-not (Test-Path $python)) {
    throw "Python venv not found at $python"
}
if (-not (Test-Path $envFile)) {
    throw ".env not found. Run scripts\write_env_from_shell.ps1 in the terminal where you set the model variables first."
}

Set-Location $projectRoot
$args = @("-m", "uvicorn", "app.main:app", "--host", $BindHost, "--port", "$Port")
if ($Reload) {
    $args += "--reload"
}
& $python @args

