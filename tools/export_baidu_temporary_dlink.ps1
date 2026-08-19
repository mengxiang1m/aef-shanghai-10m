param(
    [string]$CloudPath,
    [string]$LocalDirectory,
    [string]$CloudFolder = "/AEF_STP_temporal_2024",
    [Parameter(Mandatory = $true)]
    [string]$OutputPath
)

$ErrorActionPreference = "Stop"
if ([string]::IsNullOrWhiteSpace($CloudPath) -eq [string]::IsNullOrWhiteSpace($LocalDirectory)) {
    throw "Provide exactly one of -CloudPath or -LocalDirectory"
}
$sqlite = Get-Command sqlite3 -ErrorAction Stop
$cookieDb = Join-Path $env:APPDATA "baidunetdisk\Network\Cookies"
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
$bduss = [string]$cookieMap["BDUSS"]
if (-not $bduss) { throw "The official Baidu client is not signed in" }

$headers = @{
    Cookie = $accountCookie
    Referer = "https://pan.baidu.com/disk/main"
    "User-Agent" = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    "X-Requested-With" = "XMLHttpRequest"
}
$fields = [uri]::EscapeDataString('["uk"]')
$variablesUrl = "https://pan.baidu.com/api/gettemplatevariable?fields=$fields&clienttype=0&app_id=250528&web=1"
$variables = Invoke-RestMethod -Uri $variablesUrl -Headers $headers `
    -NoProxy -SkipCertificateCheck -HttpVersion 1.1
if ($variables.errno -ne 0 -or -not $variables.result.uk) {
    throw "Could not read the signed-in Baidu account ID"
}
$accountUk = [string]$variables.result.uk

function ConvertTo-Hex([byte[]]$Bytes, [switch]$Uppercase) {
    $hex = -join ($Bytes | ForEach-Object { $_.ToString("x2") })
    if ($Uppercase) { return $hex.ToUpperInvariant() }
    return $hex
}

$encoding = [Text.Encoding]::UTF8
$sha1 = [Security.Cryptography.SHA1]::Create()
$md5 = [Security.Cryptography.MD5]::Create()
$devUid = (ConvertTo-Hex ($md5.ComputeHash($encoding.GetBytes($bduss))) -Uppercase) + "|0"
$bdussSha1 = ConvertTo-Hex ($sha1.ComputeHash($encoding.GetBytes($bduss)))
$netdiskUserAgent = "netdisk;P2SP;3.0.0.8;netdisk;11.12.3;ANG-AN00;android-android;10.0;JSbridge4.4.0;jointBridge;1.1.0;"

function Get-TemporaryDlink([string]$Path) {
    $time = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()
    $salt = "ebrcUYiuxaZv2XGu7KIYKxUrqfnOfpDF"
    $randInput = $bdussSha1 + $accountUk + $salt + $time + $devUid
    $rand = ConvertTo-Hex ($sha1.ComputeHash($encoding.GetBytes($randInput)))
    $params = [ordered]@{
        ant = "1"; check_blue = "1"; es = "1"; esl = "1"; app_id = "250528"
        method = "locatedownload"; path = $Path; ver = "4.0"; clienttype = "17"
        channel = "0"; apn_id = "1_0"; freeisp = "0"; queryfree = "0"; use = "0"
        time = $time; rand = $rand; devuid = $devUid; cuid = $devUid
    }
    $query = ($params.GetEnumerator() | ForEach-Object {
        [uri]::EscapeDataString($_.Key) + "=" + [uri]::EscapeDataString([string]$_.Value)
    }) -join "&"
    $url = "https://pcs.baidu.com/rest/2.0/pcs/file?$query"
    $json = & curl.exe --noproxy "*" --ssl-no-revoke -sS -X POST -A $netdiskUserAgent `
        -H "Cookie: $accountCookie" $url
    if ($LASTEXITCODE -ne 0) { throw "PCS locate request failed for $Path" }
    $result = $json | ConvertFrom-Json
    $link = @($result.urls | Where-Object encrypt -eq 0 | ForEach-Object { [string]$_.url })[0]
    if (-not $link) {
        throw "PCS locate returned no usable node for ${Path}: $($result.error_code) $($result.error_msg)"
    }
    return $link
}

if ($CloudPath) {
    $lines = @(Get-TemporaryDlink $CloudPath)
} else {
    $root = (Resolve-Path -LiteralPath $LocalDirectory).Path
    $files = Get-ChildItem -LiteralPath $root -File -Filter "*.tif" | Where-Object Length -gt 100MB
    $lines = foreach ($file in $files) {
        $path = "$($CloudFolder.TrimEnd('/'))/$($file.Name)"
        $link = Get-TemporaryDlink $path
        "$($file.Name)`t$($file.Length)`t$link"
        Write-Host "Exported temporary URL: $($file.Name)"
    }
}
$absoluteOutput = if ([IO.Path]::IsPathRooted($OutputPath)) {
    $OutputPath
} else {
    Join-Path (Get-Location) $OutputPath
}
[IO.File]::WriteAllLines($absoluteOutput, $lines, [Text.UTF8Encoding]::new($false))
Write-Output "Temporary download manifest exported without account cookies: $OutputPath"
