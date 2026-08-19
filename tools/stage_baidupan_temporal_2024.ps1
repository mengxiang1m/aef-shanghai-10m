param(
    [Parameter(Mandatory = $true)]
    [string]$ShareUrl,
    [Parameter(Mandatory = $true)]
    [string]$Password,
    [string]$CloudStageFolder = "/AEF_STP_temporal_2024",
    [switch]$InventoryOnly,
    [string]$LocalVerifyRoot,
    [switch]$VerifyLocalOnly
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$shareMatch = [regex]::Match($ShareUrl, "/s/1([^?&#/]+)")
if (-not $shareMatch.Success) {
    throw "Cannot parse Baidu share URL: $ShareUrl"
}
$shortCode = $shareMatch.Groups[1].Value

$sqlite = Get-Command sqlite3 -ErrorAction SilentlyContinue
if (-not $sqlite) {
    throw "sqlite3 is required to read the signed-in official Baidu client session"
}
$cookieDb = Join-Path $env:APPDATA "baidunetdisk\Network\Cookies"
if (-not (Test-Path -LiteralPath $cookieDb)) {
    throw "Official Baidu client cookie database not found: $cookieDb"
}

$cookieRows = & $sqlite.Source -readonly -separator "`t" $cookieDb `
    "SELECT name,value FROM cookies WHERE host_key IN ('.baidu.com','.pan.baidu.com','pan.baidu.com') AND value<>'';"
$cookiePairs = foreach ($row in $cookieRows) {
    $parts = $row -split "`t", 2
    if ($parts.Count -eq 2) {
        "$($parts[0])=$($parts[1])"
    }
}
$accountCookie = $cookiePairs -join "; "
if ($accountCookie -notmatch "BDUSS=") {
    throw "The official Baidu client is not signed in"
}

$accountHeaders = @{
    Cookie = $accountCookie
    Referer = "https://pan.baidu.com/disk/main"
    "User-Agent" = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    "X-Requested-With" = "XMLHttpRequest"
}

$fields = [uri]::EscapeDataString('["bdstoken","uk"]')
$variablesUrl = "https://pan.baidu.com/api/gettemplatevariable?fields=$fields&clienttype=0&app_id=250528&web=1"
$variables = Invoke-RestMethod -Uri $variablesUrl -Headers $accountHeaders `
    -NoProxy -SkipCertificateCheck -HttpVersion 1.1
if ($variables.errno -ne 0 -or -not $variables.result.bdstoken) {
    throw "Could not read the signed-in Baidu account session: errno=$($variables.errno)"
}
$bdstoken = [string]$variables.result.bdstoken

$shareSession = New-Object Microsoft.PowerShell.Commands.WebRequestSession
$shareHeaders = @{
    Cookie = $accountCookie
    Referer = "${ShareUrl}?pwd=$Password"
    "User-Agent" = $accountHeaders["User-Agent"]
    "X-Requested-With" = "XMLHttpRequest"
}
$verifyUrl = "https://pan.baidu.com/share/verify?surl=$shortCode&channel=chunlei&web=1&app_id=250528&clienttype=0"
$verify = Invoke-RestMethod -Method Post -Uri $verifyUrl -WebSession $shareSession -Headers $shareHeaders `
    -ContentType "application/x-www-form-urlencoded" -Body @{ pwd = $Password; vcode = ""; vcode_str = "" } `
    -NoProxy -SkipCertificateCheck -HttpVersion 1.1
if ($verify.errno -ne 0) {
    throw "Baidu share verification failed: errno=$($verify.errno)"
}
$shareHeaders.Cookie = "$accountCookie; BDCLND=$($verify.randsk)"

$page = Invoke-WebRequest -Uri "${ShareUrl}?pwd=$Password" -WebSession $shareSession `
    -Headers $shareHeaders -UseBasicParsing -NoProxy -SkipCertificateCheck -HttpVersion 1.1
$localsMatch = [regex]::Match(
    $page.Content,
    'locals\.mset\((\{.*?\})\);',
    [Text.RegularExpressions.RegexOptions]::Singleline
)
if (-not $localsMatch.Success) {
    throw "Could not parse share metadata"
}
$locals = $localsMatch.Groups[1].Value | ConvertFrom-Json
$shareId = [string]$locals.shareid
$shareUk = [string]$locals.share_uk

$directoryCache = @{}
function Get-RemoteDirectory([string]$Directory) {
    if ($directoryCache.ContainsKey($Directory)) {
        return $directoryCache[$Directory]
    }
    $all = @()
    $pageNumber = 1
    do {
        $encoded = [uri]::EscapeDataString($Directory)
        $url = "https://pan.baidu.com/share/list?uk=$shareUk&shareid=$shareId&order=name&desc=0&showempty=0&web=1&page=$pageNumber&num=100&dir=$encoded&channel=chunlei&app_id=250528&clienttype=0"
        $result = Invoke-RestMethod -Uri $url -WebSession $shareSession -Headers $shareHeaders `
            -NoProxy -SkipCertificateCheck -HttpVersion 1.1
        if ($result.errno -ne 0) {
            throw "Listing $Directory failed: errno=$($result.errno)"
        }
        $batch = @($result.list)
        $all += $batch
        $pageNumber++
    } while ($batch.Count -eq 100)
    $directoryCache[$Directory] = $all
    return $all
}

$selectionRules = @(
    @{ Directory = "/data/s1/2024/raster"; Pattern = '^s1_(2024\d{2})_asc171_vv_vh_angle_utm51_10m\.tif$'; Kind = "s1" },
    @{ Directory = "/data/s1/2024/raster"; Pattern = '^s1_valid_flag_(2024\d{2})_asc171_10m\.tif$'; Kind = "s1_valid" },
    @{ Directory = "/data/s1/2024/metadata"; Pattern = '^s1_metadata_(2024\d{2})_asc171_10m\.csv$'; Kind = "s1_metadata" },
    @{ Directory = "/data/s2/2024/raster"; Pattern = '^s2_(2024\d{2})_raw_utm51_10m\.tif$'; Kind = "s2" },
    @{ Directory = "/data/s2/2024/raster"; Pattern = '^s2_valid_flag_(2024\d{2})_10m\.tif$'; Kind = "s2_valid" },
    @{ Directory = "/data/s2/2024/raster"; Pattern = '^s2_cloud_suspect_flag_(2024\d{2})_10m\.tif$'; Kind = "s2_cloud" },
    @{ Directory = "/data/s2/2024/metadata"; Pattern = '^s2_metadata_(2024\d{2})_10m\.csv$'; Kind = "s2_metadata" }
)

if ($InventoryOnly) {
    foreach ($directory in @($selectionRules.Directory | Sort-Object -Unique)) {
        Write-Output "[$directory]"
        Get-RemoteDirectory $directory |
            Where-Object { $_.server_filename -match '2024\d{2}' -and $_.server_filename -match '\.(tif|csv)$' } |
            Sort-Object server_filename |
            ForEach-Object { "  $($_.server_filename)  $([math]::Round([int64]$_.size / 1MB, 2)) MB" }
    }
    return
}

$candidates = [System.Collections.Generic.List[object]]::new()
foreach ($rule in $selectionRules) {
    foreach ($item in @(Get-RemoteDirectory $rule.Directory)) {
        $match = [regex]::Match([string]$item.server_filename, $rule.Pattern)
        if ($match.Success) {
            $candidates.Add([pscustomobject]@{
                Month = $match.Groups[1].Value
                Kind = $rule.Kind
                Name = [string]$item.server_filename
                Size = [int64]$item.size
                Path = [string]$item.path
                FsId = [string]$item.fs_id
            })
        }
    }
}

$s1Months = @($candidates | Where-Object Kind -eq "s1" | ForEach-Object Month | Sort-Object -Unique)
$s2Months = @($candidates | Where-Object Kind -eq "s2" | ForEach-Object Month | Sort-Object -Unique)
$matchedMonths = @($s1Months | Where-Object { $_ -in $s2Months } | Sort-Object -Unique)
if ($s1Months.Count -lt 3 -or $s2Months.Count -lt 3) {
    throw "Insufficient temporal coverage: S1=$($s1Months.Count), S2=$($s2Months.Count)"
}

$selected = @($candidates | Sort-Object Month, Kind)
$duplicates = @($selected | Group-Object Name | Where-Object Count -gt 1)
if ($duplicates.Count -gt 0) {
    throw "Duplicate staging filenames: $($duplicates.Name -join ', ')"
}
$totalBytes = [int64](($selected | Measure-Object Size -Sum).Sum)

Write-Output "S1 months: $($s1Months -join ', ')"
Write-Output "S2 months: $($s2Months -join ', ')"
Write-Output "Matched S1/S2 months: $($matchedMonths -join ', ')"
Write-Output ("Selected: {0} files, {1:N2} GB" -f $selected.Count, ($totalBytes / 1GB))
$allMonths = @(($s1Months + $s2Months) | Sort-Object -Unique)
foreach ($month in $allMonths) {
    $monthItems = @($selected | Where-Object Month -eq $month)
    Write-Output ("  {0}: {1}" -f $month, (($monthItems | ForEach-Object Kind) -join ", "))
}

if ($LocalVerifyRoot) {
    $localRoot = [System.IO.Path]::GetFullPath($LocalVerifyRoot)
    foreach ($entry in $selected) {
        $localPath = Join-Path $localRoot $entry.Name
        if (-not (Test-Path -LiteralPath $localPath)) {
            throw "Local file is missing: $localPath"
        }
        $localSize = (Get-Item -LiteralPath $localPath).Length
        if ($localSize -ne $entry.Size) {
            throw "Local size mismatch for $($entry.Name): expected $($entry.Size), got $localSize"
        }
    }
    Write-Output ("Local size verification passed: {0} files, {1:N2} GB" -f `
        $selected.Count, ($totalBytes / 1GB))
    if ($VerifyLocalOnly) {
        return
    }
}

$quotaUrl = "https://pan.baidu.com/api/quota?checkfree=1&checkexpire=1&bdstoken=$bdstoken&channel=chunlei&web=1&app_id=250528&clienttype=0"
$quota = Invoke-RestMethod -Uri $quotaUrl -Headers $accountHeaders `
    -NoProxy -SkipCertificateCheck -HttpVersion 1.1
if ($quota.errno -eq 0 -and $quota.total) {
    $cloudFree = [int64]$quota.total - [int64]$quota.used
    Write-Output ("Cloud free: {0:N2} GB" -f ($cloudFree / 1GB))
    if ($cloudFree -lt $totalBytes) {
        throw "Not enough Baidu cloud storage for the selected temporal data"
    }
}

$createUrl = "https://pan.baidu.com/api/create?a=commit&bdstoken=$bdstoken&channel=chunlei&web=1&app_id=250528&clienttype=0"
$createResult = Invoke-RestMethod -Method Post -Uri $createUrl -Headers $accountHeaders `
    -ContentType "application/x-www-form-urlencoded" `
    -Body @{ path = $CloudStageFolder; isdir = "1"; block_list = "[]" } `
    -NoProxy -SkipCertificateCheck -HttpVersion 1.1
if ($createResult.errno -notin @(0, -8)) {
    throw "Could not create cloud staging folder ${CloudStageFolder}: errno=$($createResult.errno)"
}

function Get-CloudItems {
    $all = @()
    $pageNumber = 1
    do {
        $encoded = [uri]::EscapeDataString($CloudStageFolder)
        $url = "https://pan.baidu.com/api/list?dir=$encoded&order=name&desc=0&showempty=0&web=1&page=$pageNumber&num=1000&channel=chunlei&app_id=250528&clienttype=0"
        $result = Invoke-RestMethod -Uri $url -Headers $accountHeaders `
            -NoProxy -SkipCertificateCheck -HttpVersion 1.1
        if ($result.errno -ne 0) {
            throw "Could not list cloud staging folder: errno=$($result.errno)"
        }
        $batch = @($result.list)
        $all += $batch
        $pageNumber++
    } while ($batch.Count -eq 1000)
    return $all
}

$cloudItems = @(Get-CloudItems)
$cloudByName = @{}
foreach ($item in $cloudItems) {
    $cloudByName[[string]$item.server_filename] = $item
}

$transferUrl = "https://pan.baidu.com/share/transfer?shareid=$shareId&from=$shareUk&bdstoken=$bdstoken&channel=chunlei&web=1&app_id=250528&clienttype=0"
$index = 0
foreach ($entry in $selected) {
    $index++
    $existing = $cloudByName[$entry.Name]
    if ($null -ne $existing) {
        if ([int64]$existing.size -ne $entry.Size) {
            throw "A different file already exists in ${CloudStageFolder}: $($entry.Name)"
        }
        Write-Output "[$index/$($selected.Count)] SKIP $($entry.Name)"
        continue
    }

    $saved = $false
    $lastErrno = $null
    for ($attempt = 1; $attempt -le 5 -and -not $saved; $attempt++) {
        $result = Invoke-RestMethod -Method Post -Uri $transferUrl -Headers $shareHeaders `
            -ContentType "application/x-www-form-urlencoded; charset=UTF-8" `
            -Body @{ fsidlist = "[$($entry.FsId)]"; path = $CloudStageFolder } `
            -NoProxy -SkipCertificateCheck -HttpVersion 1.1
        $lastErrno = $result.errno
        if ($result.errno -eq 0) {
            $saved = $true
        }
        elseif ($result.errno -in @(2, 4, 12, 31034)) {
            Start-Sleep -Seconds (2 * $attempt)
        }
        else {
            break
        }
    }
    if (-not $saved) {
        throw "Cloud save failed for $($entry.Path): errno=$lastErrno"
    }
    Write-Output "[$index/$($selected.Count)] SAVED $($entry.Name)"
    Start-Sleep -Seconds 1
}

$verified = @(Get-CloudItems)
$verifiedByName = @{}
foreach ($item in $verified) {
    $verifiedByName[[string]$item.server_filename] = $item
}
foreach ($entry in $selected) {
    $item = $verifiedByName[$entry.Name]
    if ($null -eq $item -or [int64]$item.size -ne $entry.Size) {
        throw "Cloud verification failed for $($entry.Name)"
    }
}

Write-Output ("Temporal staging complete: {0} ({1} files, {2:N2} GB)" -f `
    $CloudStageFolder, $selected.Count, ($totalBytes / 1GB))
