param(
    [Parameter(Mandatory = $true)]
    [string]$ShareUrl,
    [Parameter(Mandatory = $true)]
    [string]$Password,
    [string]$Manifest = "configs/baidupan_selection.txt",
    [string]$OutputRoot = "data/raw",
    [string]$RemotePath,
    [switch]$DirectOnly,
    [switch]$ClientOnly,
    [switch]$SubmitIndividually
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$shareMatch = [regex]::Match($ShareUrl, "/s/1([^?&#/]+)")
if (-not $shareMatch.Success) {
    throw "Cannot parse Baidu share URL: $ShareUrl"
}
$shortCode = $shareMatch.Groups[1].Value
$surl = "1$shortCode"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$manifestPath = if ([IO.Path]::IsPathRooted($Manifest)) { $Manifest } else { Join-Path $projectRoot $Manifest }
$outputPath = if ([IO.Path]::IsPathRooted($OutputRoot)) { $OutputRoot } else { Join-Path $projectRoot $OutputRoot }
New-Item -ItemType Directory -Path $outputPath -Force | Out-Null

$paths = [System.Collections.Generic.List[string]]::new()
foreach ($line in Get-Content -LiteralPath $manifestPath -Encoding UTF8) {
    if ($line -match '^# Phase 2:') { break }
    $trimmed = $line.Trim()
    if ($trimmed -and -not $trimmed.StartsWith("#")) { $paths.Add($trimmed) }
}
if ($paths.Count -eq 0) { throw "No Phase 1 paths found in $manifestPath" }
if ($RemotePath) {
    $normalizedRemotePath = if ($RemotePath.StartsWith('/')) { $RemotePath } else { "/$RemotePath" }
    if (-not $paths.Contains($normalizedRemotePath)) {
        throw "RemotePath is not present in the Phase 1 manifest: $normalizedRemotePath"
    }
    $paths = [System.Collections.Generic.List[string]]::new()
    $paths.Add($normalizedRemotePath)
}

$session = New-Object Microsoft.PowerShell.Commands.WebRequestSession
$headers = @{
    Referer = "${ShareUrl}?pwd=$Password"
    "User-Agent" = "Mozilla/5.0"
    "X-Requested-With" = "XMLHttpRequest"
}
$verifyUrl = "https://pan.baidu.com/share/verify?surl=$shortCode&channel=chunlei&web=1&app_id=250528&clienttype=0"
$verify = Invoke-RestMethod -Method Post -Uri $verifyUrl -WebSession $session -Headers $headers `
    -ContentType "application/x-www-form-urlencoded" -Body "pwd=$Password&vcode=&vcode_str="
if ($verify.errno -ne 0) { throw "Baidu share verification failed: errno=$($verify.errno)" }

$page = Invoke-WebRequest -Uri "${ShareUrl}?pwd=$Password" -WebSession $session -Headers $headers -UseBasicParsing
$localsMatch = [regex]::Match(
    $page.Content,
    'locals\.mset\((\{.*?\})\);',
    [Text.RegularExpressions.RegexOptions]::Singleline
)
if (-not $localsMatch.Success) { throw "Could not parse share metadata" }
$locals = $localsMatch.Groups[1].Value | ConvertFrom-Json
$shareId = [string]$locals.shareid
$shareUk = [string]$locals.share_uk
$sekey = [uri]::UnescapeDataString($verify.randsk)

$directoryCache = @{}
function Get-RemoteDirectory([string]$directory) {
    if ($directoryCache.ContainsKey($directory)) { return $directoryCache[$directory] }
    $all = @()
    $pageNumber = 1
    do {
        $encoded = [uri]::EscapeDataString($directory)
        $url = "https://pan.baidu.com/share/list?uk=$shareUk&shareid=$shareId&order=name&desc=0&showempty=0&web=1&page=$pageNumber&num=100&dir=$encoded&channel=chunlei&app_id=250528&clienttype=0"
        $result = Invoke-RestMethod -Uri $url -WebSession $session -Headers $headers
        if ($result.errno -ne 0) { throw "Listing $directory failed: errno=$($result.errno)" }
        $batch = @($result.list)
        $all += $batch
        $pageNumber++
    } while ($batch.Count -eq 100)
    $directoryCache[$directory] = $all
    return $all
}

$remoteItems = [System.Collections.Generic.List[object]]::new()
foreach ($remotePath in $paths) {
    $directory = $remotePath.Substring(0, $remotePath.LastIndexOf('/'))
    $filename = $remotePath.Substring($remotePath.LastIndexOf('/') + 1)
    $item = @(Get-RemoteDirectory $directory | Where-Object server_filename -eq $filename)[0]
    if ($null -eq $item) { throw "Remote file not found: $remotePath" }
    $remoteItems.Add($item)
}

$totalBytes = ($remoteItems | Measure-Object size -Sum).Sum
$freeBytes = (Get-PSDrive ([IO.Path]::GetPathRoot($outputPath).TrimEnd(':\'))).Free
Write-Output ("Phase 1: {0} files, {1:N2} GB; destination free: {2:N2} GB" -f `
    $remoteItems.Count, ($totalBytes / 1GB), ($freeBytes / 1GB))
if ($freeBytes -lt $totalBytes + 1GB) { throw "Not enough free disk space" }

function Get-DownloadResponse([object]$item, [int]$encrypt) {
    $nonce = [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()
    $configUrl = "https://pan.baidu.com/share/tplconfig?surl=$surl&fields=sign%2Ctimestamp&view_mode=1&_=$nonce"
    $config = Invoke-RestMethod -Uri $configUrl -WebSession $session -Headers $headers
    $body = @{
        encrypt = [string]$encrypt
        product = "share"
        uk = $shareUk
        primaryid = $shareId
        fid_list = "[$($item.fs_id)]"
        extra = (@{ sekey = $sekey } | ConvertTo-Json -Compress)
    }
    $api = "https://pan.baidu.com/api/sharedownload?sign=$($config.data.sign)&timestamp=$($config.data.timestamp)&channel=chunlei&web=1&app_id=250528&clienttype=0"
    return Invoke-RestMethod -Method Post -Uri $api -WebSession $session -Headers $headers `
        -ContentType "application/x-www-form-urlencoded" -Body $body
}

$cookie = ($session.Cookies.GetCookies("https://pan.baidu.com") | ForEach-Object {
    "$($_.Name)=$($_.Value)"
}) -join "; "
$clientItems = [System.Collections.Generic.List[object]]::new()
$completedBytes = 0L

foreach ($item in $remoteItems) {
    $relative = $item.path -replace '^/data/', ''
    $destination = Join-Path $outputPath ($relative -replace '/', [IO.Path]::DirectorySeparatorChar)
    $parent = Split-Path -Parent $destination
    New-Item -ItemType Directory -Path $parent -Force | Out-Null
    if ((Test-Path -LiteralPath $destination) -and (Get-Item -LiteralPath $destination).Length -eq $item.size) {
        Write-Output "SKIP complete: $relative"
        $completedBytes += [int64]$item.size
        continue
    }

    if ($ClientOnly) {
        $clientItems.Add($item)
        Write-Output "QUEUE client: $relative ($([math]::Round($item.size / 1MB, 2)) MB)"
        continue
    }

    $response = Get-DownloadResponse $item 0
    if ($response.errno -ne 0) { throw "Dlink request failed for ${relative}: errno=$($response.errno)" }
    if ($response.list -is [string]) {
        $clientItems.Add($item)
        Write-Output "QUEUE client: $relative ($([math]::Round($item.size / 1MB, 2)) MB)"
        continue
    }

    $downloadItem = @($response.list)[0]
    $part = "$destination.part"
    Write-Output "DOWNLOAD direct: $relative ($([math]::Round($item.size / 1MB, 2)) MB)"
    $downloaded = $false
    for ($attempt = 1; $attempt -le 3 -and -not $downloaded; $attempt++) {
        if ($attempt -gt 1) {
            $response = Get-DownloadResponse $item 0
            $downloadItem = @($response.list)[0]
        }
        & curl.exe -L --fail --silent --show-error --retry 4 --retry-delay 3 --continue-at - `
            --output $part -A "netdisk;P2SP;3.0.20.10" -H "Referer: https://pan.baidu.com/" `
            -H "Cookie: $cookie" $downloadItem.dlink
        $downloaded = ($LASTEXITCODE -eq 0) -and (Test-Path -LiteralPath $part) `
            -and ((Get-Item -LiteralPath $part).Length -eq $item.size)
    }
    if (-not $downloaded) { throw "Direct download incomplete: $relative" }
    Move-Item -LiteralPath $part -Destination $destination -Force
    $completedBytes += [int64]$item.size
    Write-Output ("DONE direct: {0} ({1:N2}/{2:N2} GB)" -f $relative, ($completedBytes / 1GB), ($totalBytes / 1GB))
}

function Submit-ClientItems([object[]]$items) {
    if ($items.Count -eq 0) { return }
    $ids = "[" + (($items | ForEach-Object { [string]$_.fs_id }) -join ",") + "]"
    $nonce = [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()
    $configUrl = "https://pan.baidu.com/share/tplconfig?surl=$surl&fields=sign%2Ctimestamp&view_mode=1&_=$nonce"
    $config = Invoke-RestMethod -Uri $configUrl -WebSession $session -Headers $headers
    $body = @{
        encrypt = "1"
        product = "share"
        uk = $shareUk
        primaryid = $shareId
        fid_list = $ids
        extra = (@{ sekey = $sekey } | ConvertTo-Json -Compress)
    }
    $api = "https://pan.baidu.com/api/sharedownload?sign=$($config.data.sign)&timestamp=$($config.data.timestamp)&channel=chunlei&web=1&app_id=250528&clienttype=0"
    $response = Invoke-RestMethod -Method Post -Uri $api -WebSession $session -Headers $headers `
        -ContentType "application/x-www-form-urlencoded" -Body $body
    if ($response.errno -ne 0 -or $response.list -isnot [string]) {
        throw "Could not prepare Baidu client task: errno=$($response.errno)"
    }
    $localUrl = "https://localhost.pan.baidu.com:10000/guanjia?method=DownloadShareItems&uk=0&checkuser=0"
    $localHeaders = @{ Origin = "https://pan.baidu.com"; Referer = "https://pan.baidu.com/" }
    $localResult = Invoke-RestMethod -Method Post -Uri $localUrl -SkipCertificateCheck `
        -Headers $localHeaders -ContentType "application/x-www-form-urlencoded" `
        -Body @{ filelist = $response.list }
    Write-Output "Baidu client response: $($localResult.info)"
}

if ($clientItems.Count -gt 0) {
    if ($DirectOnly) {
        Write-Output "$($clientItems.Count) large files require the installed Baidu client; not submitted because -DirectOnly was set."
        exit 2
    }
    if ($SubmitIndividually) {
        foreach ($clientItem in $clientItems) { Submit-ClientItems @($clientItem) }
    } else {
        Submit-ClientItems @($clientItems)
    }
    Write-Output "$($clientItems.Count) large files were submitted to the installed Baidu client."
}
