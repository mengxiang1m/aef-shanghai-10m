param(
    [string]$LocalRoot = "D:\AEF\aef_shanghai\data\raw\AEF_STP_temporal_2024",
    [string]$RemoteHost = "DM02",
    [string]$RemoteRoot = "/home/zhaoqing/aef_shanghai/data/raw/AEF_STP_temporal_2024"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $LocalRoot -PathType Container)) {
    throw "Local temporal data directory does not exist: $LocalRoot"
}

& ssh -n -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=15 -o ServerAliveCountMax=2 $RemoteHost "mkdir -p '$RemoteRoot'"
if ($LASTEXITCODE -ne 0) {
    throw "Could not create remote directory: ${RemoteHost}:$RemoteRoot"
}

$files = Get-ChildItem -LiteralPath $LocalRoot -File | Sort-Object `
    @{ Expression = { if ($_.Length -lt 100MB) { 0 } elseif ($_.Name -like 's2_*') { 1 } else { 2 } } }, `
    Name
$totalBytes = ($files | Measure-Object -Property Length -Sum).Sum
$completedBytes = 0L
$remoteInventory = @{}
& ssh -n -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=15 -o ServerAliveCountMax=2 $RemoteHost `
    "find '$RemoteRoot' -maxdepth 1 -type f -printf '%f %s\n'" | ForEach-Object {
        $parts = $_ -split ' ', 2
        if ($parts.Count -eq 2) {
            $remoteInventory[$parts[0]] = [long]$parts[1]
        }
    }

foreach ($file in $files) {
    $remotePath = "$RemoteRoot/$($file.Name)"
    $remoteSize = if ($remoteInventory.ContainsKey($file.Name)) { $remoteInventory[$file.Name] } else { 0L }

    if ($remoteSize -eq $file.Length) {
        $completedBytes += $file.Length
        $percent = 100.0 * $completedBytes / $totalBytes
        Write-Output ("SKIP {0} ({1:N2}% complete)" -f $file.Name, $percent)
        continue
    }

    Write-Output ("COPY {0} ({1:N2} GiB)" -f $file.Name, ($file.Length / 1GB))
    & scp -q -C -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=15 -o ServerAliveCountMax=2 -- `
        $file.FullName "${RemoteHost}:$remotePath"
    if ($LASTEXITCODE -ne 0) {
        throw "scp failed for $($file.FullName)"
    }

    $completedBytes += $file.Length
    $percent = 100.0 * $completedBytes / $totalBytes
    Write-Output ("DONE {0} ({1:N2}% complete)" -f $file.Name, $percent)
}

Write-Output "VERIFY final remote inventory"
$finalInventory = @{}
& ssh -n -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=15 -o ServerAliveCountMax=2 $RemoteHost `
    "find '$RemoteRoot' -maxdepth 1 -type f -printf '%f %s\n'" | ForEach-Object {
        $parts = $_ -split ' ', 2
        if ($parts.Count -eq 2) {
            $finalInventory[$parts[0]] = [long]$parts[1]
        }
    }
foreach ($file in $files) {
    if (-not $finalInventory.ContainsKey($file.Name) -or $finalInventory[$file.Name] -ne $file.Length) {
        throw "Final remote size mismatch: $($file.Name)"
    }
}
Write-Output ("SYNC_COMPLETE files={0} bytes={1}" -f $files.Count, $totalBytes)
