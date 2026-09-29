param(
    [Parameter(Mandatory = $true)]
    [string]$ApiBase,

    [Parameter(Mandatory = $true)]
    [string]$ApiKey,

    [string]$Model = "deepseek-v4.1-flash",
    [int]$Port = 8031,
    [int]$Workers = 8,
    [int]$TopK = 100,
    [string]$ArtifactRoot = "D:\workspace\memory\amc",
    [string]$Database = "D:\workspace\memory\amc\longmemeval_s_deepseek.db",
    [string]$Dataset = "D:\workspace\memory\aml-text-memory\examples\longmemeval_s_eval.jsonl"
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "Python venv not found at $python"
}
if (-not (Test-Path $Dataset)) {
    throw "Dataset not found at $Dataset"
}
if (-not (Test-Path $ArtifactRoot)) {
    New-Item -ItemType Directory -Path $ArtifactRoot | Out-Null
}

# DeepSeek is used only for local experiments.  Embedding stays on the
# official text-embedding-v4 configuration from .env so vectors remain
# comparable with the compliant pipeline.
$env:AML_AUTH_MODE = "none"
$env:AML_DATABASE_PATH = $Database
$env:AML_EMBEDDING_PROVIDER = "openai"
$env:AML_EMBEDDING_MODEL = "text-embedding-v4"
# The decomposer has priority when enabled.  Keep the organizer on the
# deterministic rule path so only one LLM extraction strategy is active.
$env:AML_ORGANIZER_PROVIDER = "rule"
$env:AML_DECOMPOSER_PROVIDER = "deepseek"
$env:AML_DECOMPOSER_MODEL = $Model
$env:AML_DECOMPOSER_API_BASE = $ApiBase
$env:AML_DECOMPOSER_API_KEY = $ApiKey
$env:AML_RERANKER_PROVIDER = "lexical"
$env:AML_RETRIEVAL_CANDIDATE_K = "10000"
$env:AML_MAX_RETURN_ITEMS = "100"
$env:AML_MAX_CONTENT_CHARS = "4000"

$serverOut = Join-Path $ArtifactRoot "deepseek_server_out.log"
$serverErr = Join-Path $ArtifactRoot "deepseek_server_err.log"
$evalOut = Join-Path $ArtifactRoot "deepseek_eval.log"
$evalErr = Join-Path $ArtifactRoot "deepseek_eval.err.log"
$report = Join-Path $ArtifactRoot "longmemeval_s_deepseek_report.json"

Remove-Item $serverOut, $serverErr, $evalOut, $evalErr -ErrorAction SilentlyContinue

$serverArgs = @(
    "-m", "uvicorn", "app.main:app",
    "--host", "127.0.0.1",
    "--port", "$Port"
)
$server = Start-Process -FilePath $python -ArgumentList $serverArgs -WorkingDirectory $projectRoot -RedirectStandardOutput $serverOut -RedirectStandardError $serverErr -WindowStyle Hidden -PassThru
Set-Content -Path (Join-Path $ArtifactRoot "deepseek_server_pid.txt") -Value $server.Id -Encoding ascii

$healthy = $false
for ($i = 0; $i -lt 120; $i++) {
    try {
        $health = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 2
        if ($health.status -eq "ok") {
            $healthy = $true
            break
        }
    } catch {
    }
    Start-Sleep -Milliseconds 1000
}
if (-not $healthy) {
    throw "DeepSeek API server did not become healthy. See $serverErr"
}

$evalArgs = @(
    (Join-Path $projectRoot "scripts\eval_retrieval_parallel.py"),
    "--base-url", "http://127.0.0.1:$Port",
    "--dataset", $Dataset,
    "--top-k", "$TopK",
    "--workers", "$Workers",
    "--output", $report
)
$eval = Start-Process -FilePath $python -ArgumentList $evalArgs -WorkingDirectory $projectRoot -RedirectStandardOutput $evalOut -RedirectStandardError $evalErr -WindowStyle Hidden -PassThru
Set-Content -Path (Join-Path $ArtifactRoot "deepseek_eval_pid.txt") -Value $eval.Id -Encoding ascii

Write-Output "DeepSeek server started: pid=$($server.Id) port=$Port"
Write-Output "DeepSeek eval started:   pid=$($eval.Id) workers=$Workers"
Write-Output "Server log: $serverOut"
Write-Output "Eval log:   $evalOut"
Write-Output "Report:     $report"
