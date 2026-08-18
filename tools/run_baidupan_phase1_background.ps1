$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$logDirectory = Join-Path $projectRoot "data\logs"
$stdoutLog = Join-Path $logDirectory "baidupan_phase1.out.log"
$stderrLog = Join-Path $logDirectory "baidupan_phase1.err.log"
$downloader = Join-Path $PSScriptRoot "download_baidupan_phase1_authenticated.ps1"

New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null
"=== background run started: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ===" | Add-Content -LiteralPath $stdoutLog

try {
    & $downloader `
        -ShareUrl "https://pan.baidu.com/s/1uy95FpHhUYGNg7nvd3sT6w" `
        -Password "9yj8" `
        1>> $stdoutLog 2>> $stderrLog
    "=== background run completed: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ===" | Add-Content -LiteralPath $stdoutLog
} catch {
    "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $($_ | Out-String)" | Add-Content -LiteralPath $stderrLog
    exit 1
}
