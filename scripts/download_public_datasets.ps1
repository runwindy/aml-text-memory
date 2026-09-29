$ErrorActionPreference = "Continue"
$base = "D:\workspace\memory\datasets"
New-Item -ItemType Directory -Force -Path $base | Out-Null

$repos = @{
  "longmemeval" = "https://github.com/xiaowu0162/LongMemEval.git"
  "clbench" = "https://github.com/Tencent-Hunyuan/CL-bench.git"
  "personamem-v2" = "https://github.com/bowen-upenn/PersonaMem-v2.git"
  "beam" = "https://github.com/mohammadtavakoli78/BEAM.git"
  "scriptmem" = "https://github.com/memorax-ai/ScriptMem.git"
}

foreach ($name in $repos.Keys) {
  $target = Join-Path $base $name
  if (Test-Path $target) {
    Write-Output "SKIP $name already exists"
    continue
  }
  Write-Output "CLONE $name"
  git clone --depth 1 --single-branch $repos[$name] $target 2>&1 | ForEach-Object { Write-Output $_ }
  if (Test-Path $target) {
    $size = (Get-ChildItem -Recurse $target | Measure-Object -Property Length -Sum).Sum
    Write-Output "DONE $name size_mb=$([math]::Round($size/1MB,1))"
  } else {
    Write-Output "FAILED $name"
  }
}
Write-Output "ALL_DONE"
