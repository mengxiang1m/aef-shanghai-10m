param(
    [Parameter(Mandatory = $true)]
    [string]$ShareUrl,
    [Parameter(Mandatory = $true)]
    [string]$Password,
    [string]$Manifest = "configs/baidupan_selection.txt",
    [string]$OutputRoot = "data/raw",
    [string]$CloudStageFolder = "/AEF_shanghai_phase1"
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

function Get-Rc4Base64([string]$key, [string]$value) {
    if (-not $key) { throw "Empty Baidu sign key" }
    $a = New-Object int[] 256
    $p = New-Object int[] 256
    for ($q = 0; $q -lt 256; $q++) {
        $a[$q] = [int][char]$key[$q % $key.Length]
        $p[$q] = $q
    }
    $u = 0
    for ($q = 0; $q -lt 256; $q++) {
        $u = ($u + $p[$q] + $a[$q]) % 256
        $tmp = $p[$q]; $p[$q] = $p[$u]; $p[$u] = $tmp
    }
    $bytes = New-Object byte[] $value.Length
    $i = 0; $u = 0
    for ($q = 0; $q -lt $value.Length; $q++) {
        $i = ($i + 1) % 256
        $u = ($u + $p[$i]) % 256
        $tmp = $p[$i]; $p[$i] = $p[$u]; $p[$u] = $tmp
        $k = $p[($p[$i] + $p[$u]) % 256]
        $bytes[$q] = ([int][char]$value[$q] -bxor $k)
    }
    return [Convert]::ToBase64String($bytes)
}

$shareMatch = [regex]::Match($ShareUrl, "/s/1([^?&#/]+)")
if (-not $shareMatch.Success) { throw "Cannot parse Baidu share URL: $ShareUrl" }
$shortCode = $shareMatch.Groups[1].Value
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
$duplicates = $paths | ForEach-Object { Split-Path $_ -Leaf } | Group-Object | Where-Object Count -gt 1
if ($duplicates) { throw "The cloud staging folder requires unique filenames; duplicate: $($duplicates.Name -join ', ')" }

$sqlite = Get-Command sqlite3 -ErrorAction SilentlyContinue
if (-not $sqlite) { throw "sqlite3 is required to read the signed-in official Baidu client session" }
$cookieDb = Join-Path $env:APPDATA "baidunetdisk\Network\Cookies"
if (-not (Test-Path -LiteralPath $cookieDb)) { throw "Official Baidu client cookie database not found: $cookieDb" }
$cookieRows = & $sqlite.Source -readonly -separator "`t" $cookieDb `
    "SELECT name,value FROM cookies WHERE host_key IN ('.baidu.com','.pan.baidu.com','pan.baidu.com') AND value<>'';"
$cookieMap = @{}
$cookiePairs = foreach ($row in $cookieRows) {
    $parts = $row -split "`t", 2
    if ($parts.Count -eq 2) {
        $cookieMap[$parts[0]] = $parts[1]
        "$($parts[0])=$($parts[1])"
    }
}
$accountCookie = $cookiePairs -join "; "
if ($accountCookie -notmatch 'BDUSS=') { throw "The official Baidu client is not signed in" }
$bduss = [string]$cookieMap["BDUSS"]
$netdiskUserAgent = "netdisk;P2SP;3.0.0.8;netdisk;11.12.3;ANG-AN00;android-android;10.0;JSbridge4.4.0;jointBridge;1.1.0;"
$accountHeaders = @{
    Cookie = $accountCookie
    Referer = "https://pan.baidu.com/disk/main"
    "User-Agent" = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    "X-Requested-With" = "XMLHttpRequest"
}

function Get-PanVariables {
    $fields = [uri]::EscapeDataString('["bdstoken","sign1","sign3","timestamp","uk"]')
    $url = "https://pan.baidu.com/api/gettemplatevariable?fields=$fields&clienttype=0&app_id=250528&web=1"
    $result = Invoke-RestMethod -Uri $url -Headers $accountHeaders
    if ($result.errno -ne 0 -or -not $result.result.bdstoken) {
        throw "Could not read the signed-in Baidu account session: errno=$($result.errno)"
    }
    return $result.result
}

$variables = Get-PanVariables
$bdstoken = [string]$variables.bdstoken
$accountUk = [string]$variables.uk
$createUrl = "https://pan.baidu.com/api/create?a=commit&bdstoken=$bdstoken&channel=chunlei&web=1&app_id=250528&clienttype=0"
$createResult = Invoke-RestMethod -Method Post -Uri $createUrl -Headers $accountHeaders `
    -ContentType "application/x-www-form-urlencoded" `
    -Body @{ path = $CloudStageFolder; isdir = "1"; block_list = "[]" }
if ($createResult.errno -notin @(0, -8)) {
    throw "Could not create cloud staging folder ${CloudStageFolder}: errno=$($createResult.errno)"
}

$shareSession = New-Object Microsoft.PowerShell.Commands.WebRequestSession
$shareHeaders = @{
    Cookie = $accountCookie
    Referer = "${ShareUrl}?pwd=$Password"
    "User-Agent" = $accountHeaders["User-Agent"]
    "X-Requested-With" = "XMLHttpRequest"
}
$verifyUrl = "https://pan.baidu.com/share/verify?surl=$shortCode&channel=chunlei&web=1&app_id=250528&clienttype=0"
$verify = Invoke-RestMethod -Method Post -Uri $verifyUrl -WebSession $shareSession -Headers $shareHeaders `
    -ContentType "application/x-www-form-urlencoded" -Body @{ pwd = $Password; vcode = ""; vcode_str = "" }
if ($verify.errno -ne 0) { throw "Baidu share verification failed: errno=$($verify.errno)" }
$shareHeaders.Cookie = "$accountCookie; BDCLND=$($verify.randsk)"

$page = Invoke-WebRequest -Uri "${ShareUrl}?pwd=$Password" -WebSession $shareSession -Headers $shareHeaders -UseBasicParsing
$localsMatch = [regex]::Match($page.Content, 'locals\.mset\((\{.*?\})\);', [Text.RegularExpressions.RegexOptions]::Singleline)
if (-not $localsMatch.Success) { throw "Could not parse share metadata" }
$locals = $localsMatch.Groups[1].Value | ConvertFrom-Json
$shareId = [string]$locals.shareid
$shareUk = [string]$locals.share_uk

$directoryCache = @{}
function Get-RemoteDirectory([string]$directory) {
    if ($directoryCache.ContainsKey($directory)) { return $directoryCache[$directory] }
    $all = @(); $pageNumber = 1
    do {
        $encoded = [uri]::EscapeDataString($directory)
        $url = "https://pan.baidu.com/share/list?uk=$shareUk&shareid=$shareId&order=name&desc=0&showempty=0&web=1&page=$pageNumber&num=100&dir=$encoded&channel=chunlei&app_id=250528&clienttype=0"
        $result = Invoke-RestMethod -Uri $url -WebSession $shareSession -Headers $shareHeaders
        if ($result.errno -ne 0) { throw "Listing $directory failed: errno=$($result.errno)" }
        $batch = @($result.list); $all += $batch; $pageNumber++
    } while ($batch.Count -eq 100)
    $directoryCache[$directory] = $all
    return $all
}

$sourceItems = [System.Collections.Generic.List[object]]::new()
foreach ($remotePath in $paths) {
    $directory = $remotePath.Substring(0, $remotePath.LastIndexOf('/'))
    $filename = Split-Path $remotePath -Leaf
    $item = @(Get-RemoteDirectory $directory | Where-Object server_filename -eq $filename)[0]
    if ($null -eq $item) { throw "Remote file not found: $remotePath" }
    $sourceItems.Add($item)
}
$totalBytes = ($sourceItems | Measure-Object size -Sum).Sum
$driveName = [IO.Path]::GetPathRoot($outputPath).TrimEnd(':\')
$freeBytes = (Get-PSDrive $driveName).Free
Write-Output ("Phase 1 selected: {0} files, {1:N2} GB; destination free: {2:N2} GB" -f `
    $sourceItems.Count, ($totalBytes / 1GB), ($freeBytes / 1GB))
if ($freeBytes -lt $totalBytes + 1GB) { throw "Not enough free disk space" }

function Get-CloudItems {
    $encoded = [uri]::EscapeDataString($CloudStageFolder)
    $url = "https://pan.baidu.com/api/list?dir=$encoded&order=name&desc=0&showempty=0&web=1&page=1&num=1000&channel=chunlei&app_id=250528&clienttype=0"
    $result = Invoke-RestMethod -Uri $url -Headers $accountHeaders
    if ($result.errno -ne 0) { throw "Could not list cloud staging folder: errno=$($result.errno)" }
    return @($result.list)
}

function Save-ToCloud([object]$sourceItem) {
    $url = "https://pan.baidu.com/share/transfer?shareid=$shareId&from=$shareUk&bdstoken=$bdstoken&channel=chunlei&web=1&app_id=250528&clienttype=0"
    $result = Invoke-RestMethod -Method Post -Uri $url -Headers $shareHeaders `
        -ContentType "application/x-www-form-urlencoded; charset=UTF-8" `
        -Body @{ fsidlist = "[$($sourceItem.fs_id)]"; path = $CloudStageFolder }
    if ($result.errno -ne 0) {
        throw "Cloud save failed for $($sourceItem.server_filename): errno=$($result.errno)"
    }
}

function ConvertTo-Hex([byte[]]$bytes, [switch]$Uppercase) {
    $value = ($bytes | ForEach-Object { $_.ToString("x2") }) -join ""
    if ($Uppercase) { return $value.ToUpperInvariant() }
    return $value
}

function Get-LocateDlinks([string]$cloudPath) {
    $encoding = [Text.Encoding]::UTF8
    $sha1 = [Security.Cryptography.SHA1]::Create()
    $md5 = [Security.Cryptography.MD5]::Create()
    $devUid = (ConvertTo-Hex ($md5.ComputeHash($encoding.GetBytes($bduss))) -Uppercase) + "|0"
    $bdussSha1 = ConvertTo-Hex ($sha1.ComputeHash($encoding.GetBytes($bduss)))
    $time = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()
    $salt = "ebrcUYiuxaZv2XGu7KIYKxUrqfnOfpDF"
    $randInput = $bdussSha1 + $accountUk + $salt + $time + $devUid
    $rand = ConvertTo-Hex ($sha1.ComputeHash($encoding.GetBytes($randInput)))
    $params = [ordered]@{
        ant = "1"; check_blue = "1"; es = "1"; esl = "1"; app_id = "250528"
        method = "locatedownload"; path = $cloudPath; ver = "4.0"; clienttype = "17"
        channel = "0"; apn_id = "1_0"; freeisp = "0"; queryfree = "0"; use = "0"
        time = $time; rand = $rand; devuid = $devUid; cuid = $devUid
    }
    $query = ($params.GetEnumerator() | ForEach-Object {
        [uri]::EscapeDataString($_.Key) + "=" + [uri]::EscapeDataString([string]$_.Value)
    }) -join "&"
    $url = "https://pcs.baidu.com/rest/2.0/pcs/file?$query"
    $json = & curl.exe -sS -X POST -A $netdiskUserAgent -H "Cookie: $accountCookie" $url
    if ($LASTEXITCODE -ne 0) { throw "PCS locate request failed for $cloudPath" }
    $result = $json | ConvertFrom-Json
    $links = @($result.urls | Where-Object encrypt -eq 0 | ForEach-Object { [string]$_.url })
    if ($links.Count -eq 0) {
        throw "PCS locate returned no usable node for ${cloudPath}: $($result.error_code) $($result.error_msg)"
    }
    return $links
}

function Invoke-SegmentedDownload([object]$cloudItem, [string]$partPath, [int64]$expectedSize) {
    $prefixLength = if (Test-Path -LiteralPath $partPath) { (Get-Item -LiteralPath $partPath).Length } else { 0L }
    if ($prefixLength -gt $expectedSize) { throw "Partial file is larger than expected: $partPath" }
    if ($prefixLength -eq $expectedSize) { return $true }

    $remaining = $expectedSize - $prefixLength
    # The PCS nodes reject bounded ranges above a few MB. Use 4 MB pieces and
    # run a moderate number concurrently.
    $segmentSize = [int64](4MB)
    $segmentCount = [int][Math]::Ceiling($remaining / $segmentSize)
    $maxParallel = 1
    $links = @(Get-LocateDlinks ([string]$cloudItem.path))
    $curlPath = (Get-Command curl.exe).Source
    $segments = [System.Collections.Generic.List[object]]::new()

    for ($segment = 0; $segment -lt $segmentCount; $segment++) {
        $start = $prefixLength + [int64]$segment * $segmentSize
        if ($start -ge $expectedSize) { break }
        $end = [Math]::Min($expectedSize - 1, $start + $segmentSize - 1)
        $segmentPath = "${partPath}.seg$($segment.ToString('D2'))"
        $segmentExpected = $end - $start + 1
        if ((Test-Path -LiteralPath $segmentPath) -and (Get-Item -LiteralPath $segmentPath).Length -ne $segmentExpected) {
            Remove-Item -LiteralPath $segmentPath -Force
        }
        if ((Test-Path -LiteralPath $segmentPath) -and (Get-Item -LiteralPath $segmentPath).Length -eq $segmentExpected) {
            $segments.Add([pscustomobject]@{ Index = $segment; Start = $start; End = $end; Path = $segmentPath; Expected = $segmentExpected; Complete = $true })
        } else {
            $segments.Add([pscustomobject]@{ Index = $segment; Start = $start; End = $end; Path = $segmentPath; Expected = $segmentExpected; Complete = $false })
        }
    }

    $pending = @($segments | Where-Object { -not $_.Complete })
    for ($batchStart = 0; $batchStart -lt $pending.Count; $batchStart += $maxParallel) {
        $batchEnd = [Math]::Min($pending.Count - 1, $batchStart + $maxParallel - 1)
        $batch = [System.Collections.Generic.List[object]]::new()
        for ($position = $batchStart; $position -le $batchEnd; $position++) {
            $segmentInfo = $pending[$position]
            # The first PCS node is range-verified; alternate nodes are often stale.
            $arguments = @(
                "-L", "--fail", "--silent", "--show-error", "--retry", "4", "--retry-delay", "2",
                "--range", "$($segmentInfo.Start)-$($segmentInfo.End)", "--output", $segmentInfo.Path,
                "-A", $netdiskUserAgent, "-H", "Cookie: $accountCookie",
                "-H", "Referer: https://pan.baidu.com/", $links[0]
            )
            $startInfo = [Diagnostics.ProcessStartInfo]::new()
            $startInfo.FileName = $curlPath
            $startInfo.UseShellExecute = $false
            $startInfo.CreateNoWindow = $true
            $startInfo.RedirectStandardError = $true
            foreach ($argument in $arguments) { $startInfo.ArgumentList.Add($argument) }
            $process = [Diagnostics.Process]::Start($startInfo)
            $batch.Add([pscustomobject]@{ Process = $process; Info = $segmentInfo })
        }

        $failed = $null
        do {
            $running = @($batch | Where-Object { -not $_.Process.HasExited })
            $failed = @($batch | Where-Object { $_.Process.HasExited -and $_.Process.ExitCode -ne 0 })[0]
            if ($failed) {
                foreach ($entry in $running) { $entry.Process.Kill($true) }
                break
            }
            if ($running.Count -gt 0) { Start-Sleep -Milliseconds 500 }
        } while ($running.Count -gt 0)

        foreach ($entry in $batch) {
            $entry.Process.WaitForExit()
            $actual = if (Test-Path -LiteralPath $entry.Info.Path) { (Get-Item -LiteralPath $entry.Info.Path).Length } else { 0 }
            if ($entry.Process.ExitCode -ne 0 -or $actual -ne $entry.Info.Expected) {
                $detail = $entry.Process.StandardError.ReadToEnd().Trim()
                throw "Segment download incomplete: $($entry.Info.Path) (expected $($entry.Info.Expected), got $actual). $detail"
            }
        }
        $completedPieces = [Math]::Min($batchEnd + 1, $pending.Count)
        Write-Output "    range pieces: $completedPieces/$($pending.Count)"
    }

    $output = [IO.File]::Open($partPath, [IO.FileMode]::Append, [IO.FileAccess]::Write, [IO.FileShare]::None)
    try {
        foreach ($segmentInfo in $segments) {
            $input = [IO.File]::OpenRead($segmentInfo.Path)
            try { $input.CopyTo($output) } finally { $input.Dispose() }
        }
    } finally {
        $output.Dispose()
    }
    foreach ($segmentInfo in $segments) { Remove-Item -LiteralPath $segmentInfo.Path -Force }
    return (Get-Item -LiteralPath $partPath).Length -eq $expectedSize
}

$doneBytes = 0L; $index = 0
foreach ($sourceItem in $sourceItems) {
    $index++
    $relative = $sourceItem.path -replace '^/data/', ''
    $destination = Join-Path $outputPath ($relative -replace '/', [IO.Path]::DirectorySeparatorChar)
    if ((Test-Path -LiteralPath $destination) -and (Get-Item -LiteralPath $destination).Length -eq $sourceItem.size) {
        $doneBytes += [int64]$sourceItem.size
        Write-Output "[$index/$($sourceItems.Count)] SKIP complete: $relative"
        continue
    }

    $cloudItem = @(Get-CloudItems | Where-Object server_filename -eq $sourceItem.server_filename)[0]
    if ($cloudItem -and $cloudItem.size -ne $sourceItem.size) {
        throw "A different file with the same name already exists in ${CloudStageFolder}: $($sourceItem.server_filename)"
    }
    if (-not $cloudItem) {
        Write-Output "[$index/$($sourceItems.Count)] SAVE cloud: $relative"
        Save-ToCloud $sourceItem
        $cloudItem = @(Get-CloudItems | Where-Object server_filename -eq $sourceItem.server_filename)[0]
        if (-not $cloudItem -or $cloudItem.size -ne $sourceItem.size) {
            throw "Cloud save verification failed: $relative"
        }
    }

    $parent = Split-Path -Parent $destination
    New-Item -ItemType Directory -Path $parent -Force | Out-Null
    $part = "$destination.part"
    Write-Output "[$index/$($sourceItems.Count)] DOWNLOAD: $relative ($([math]::Round($sourceItem.size / 1MB, 2)) MB)"
    $downloaded = $false
    if ($sourceItem.size -ge 128MB) {
        $downloaded = Invoke-SegmentedDownload $cloudItem $part ([int64]$sourceItem.size)
    }
    for ($attempt = 1; $attempt -le 3 -and -not $downloaded; $attempt++) {
        $dlinks = @(Get-LocateDlinks ([string]$cloudItem.path))
        $dlink = $dlinks[($attempt - 1) % $dlinks.Count]
        $resumeArgs = if ((Test-Path -LiteralPath $part) -and (Get-Item -LiteralPath $part).Length -gt 0) {
            @("--continue-at", "-")
        } else {
            @("--range", "0-")
        }
        & curl.exe -L --fail --silent --show-error --retry 4 --retry-delay 2 @resumeArgs `
            --output $part -A $netdiskUserAgent -H "Cookie: $accountCookie" `
            -H "Referer: https://pan.baidu.com/" $dlink
        $downloaded = ($LASTEXITCODE -eq 0) -and (Test-Path -LiteralPath $part) `
            -and ((Get-Item -LiteralPath $part).Length -eq $sourceItem.size)
    }
    if (-not $downloaded) {
        $actual = if (Test-Path -LiteralPath $part) { (Get-Item -LiteralPath $part).Length } else { 0 }
        throw "Download incomplete: $relative (expected $($sourceItem.size), got $actual)"
    }
    Move-Item -LiteralPath $part -Destination $destination -Force
    $doneBytes += [int64]$sourceItem.size
    Write-Output ("[$index/$($sourceItems.Count)] DONE: {0} ({1:N2}/{2:N2} GB)" -f `
        $relative, ($doneBytes / 1GB), ($totalBytes / 1GB))
}

Write-Output "Phase 1 download complete and size-verified: $outputPath"
