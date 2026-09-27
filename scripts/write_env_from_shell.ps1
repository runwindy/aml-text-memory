param(
    [string]$OutputPath = "",
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
if (-not $OutputPath) {
    $OutputPath = Join-Path $projectRoot ".env"
}

$names = @(
    "AML_EMBEDDING_PROVIDER",
    "AML_EMBEDDING_MODEL",
    "AML_EMBEDDING_API_BASE",
    "AML_EMBEDDING_API_KEY",
    "AML_EMBEDDING_BATCH_SIZE",
    "AML_EMBEDDING_DIM",
    "AML_ORGANIZER_PROVIDER",
    "AML_ORGANIZER_MODEL",
    "AML_ORGANIZER_API_BASE",
    "AML_ORGANIZER_API_KEY",
    "AML_ORGANIZER_MAX_TOKENS",
    "AML_ORGANIZER_TIMEOUT",
    "AML_DATABASE_PATH",
    "AML_AUTH_MODE",
    "AML_MEMORY_SYSTEM_KEY",
    "AML_RETRIEVAL_CANDIDATE_K",
    "AML_MAX_RETURN_ITEMS",
    "AML_MAX_CONTENT_CHARS"
)

if ((Test-Path $OutputPath) -and -not $Force) {
    throw "$OutputPath already exists. Re-run with -Force to overwrite."
}

$defaults = @{
    "AML_DATABASE_PATH" = "./data/aml_v4.db"
    "AML_AUTH_MODE" = "none"
    "AML_RETRIEVAL_CANDIDATE_K" = "10000"
    "AML_MAX_RETURN_ITEMS" = "100"
    "AML_MAX_CONTENT_CHARS" = "4000"
    "AML_EMBEDDING_BATCH_SIZE" = "10"
    "AML_ORGANIZER_MAX_TOKENS" = "1024"
    "AML_ORGANIZER_TIMEOUT" = "120"
}

$lines = @()
$summary = @()
foreach ($name in $names) {
    $value = [Environment]::GetEnvironmentVariable($name, "Process")
    if ([string]::IsNullOrWhiteSpace($value) -and $defaults.ContainsKey($name)) {
        $value = $defaults[$name]
    }
    if ([string]::IsNullOrWhiteSpace($value)) {
        $summary += "$name = <missing>"
        continue
    }

    # .env values are simple key=value lines. Reject obvious unsafe line breaks.
    $clean = $value.Trim()
    if ($clean.Contains("`n") -or $clean.Contains("`r")) {
        throw "$name contains a line break and cannot be written safely."
    }
    $lines += "$name=$clean"

    if ($name -match "KEY|SECRET|TOKEN|PASSWORD") {
        $summary += "$name = <set length=$($clean.Length)>"
    } else {
        $summary += "$name = $clean"
    }
}

$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllText($OutputPath, ($lines -join "`n") + "`n", $utf8NoBom)

Write-Host "Wrote $OutputPath"
$summary | ForEach-Object { Write-Host $_ }
