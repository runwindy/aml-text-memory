$base = "D:\workspace\memory\datasets"
New-Item -ItemType Directory -Force -Path $base | Out-Null
$repos = @(
  @{ name="longmemeval"; url="https://codeload.github.com/xiaowu0162/LongMemEval/zip/refs/heads/main" },
  @{ name="personamem-v2"; url="https://codeload.github.com/bowen-upenn/PersonaMem-v2/zip/refs/heads/main" },
  @{ name="beam"; url="https://codeload.github.com/mohammadtavakoli78/BEAM/zip/refs/heads/main" },
  @{ name="clbench"; url="https://codeload.github.com/Tencent-Hunyuan/CL-bench/zip/refs/heads/main" }
)
foreach ($repo in $repos) {
  $out = Join-Path $base ($repo.name + ".zip")
  if (Test-Path $out) { Write-Output "SKIP $($repo.name)"; continue }
  Write-Output "DOWNLOAD $($repo.name)"
  curl.exe -L --fail --max-time 900 -o $out $repo.url 2>&1 | ForEach-Object { Write-Output $_ }
  if (Test-Path $out) {
    Write-Output "DONE $($repo.name) size_mb=$([math]::Round((Get-Item $out).Length/1MB,2))"
  } else {
    Write-Output "FAILED $($repo.name)"
  }
}
Write-Output "ALL_DONE"
