param(
    [string]$PartsDirectory = ".",
    [string]$OutputPath = "aef_shanghai_pretrain_best.pt",
    [string]$ExpectedSha256 = "c0746d3c2ea71d5d5b34f19dcbf56780055db8719d4fa121cf23e4c34b13578f"
)

$ErrorActionPreference = "Stop"
$parts = @(Get-ChildItem -LiteralPath $PartsDirectory -File |
    Where-Object { $_.Name -match '^aef_shanghai_pretrain_best\.pt\.part\d{3}$' } |
    Sort-Object Name)

if ($parts.Count -eq 0) {
    throw "No checkpoint parts were found in $PartsDirectory"
}

$resolvedOutput = [System.IO.Path]::GetFullPath($OutputPath)
$output = [System.IO.File]::Create($resolvedOutput)
try {
    foreach ($part in $parts) {
        $input = [System.IO.File]::OpenRead($part.FullName)
        try {
            $input.CopyTo($output)
        }
        finally {
            $input.Dispose()
        }
    }
}
finally {
    $output.Dispose()
}

$actualSha256 = (Get-FileHash -LiteralPath $resolvedOutput -Algorithm SHA256).Hash.ToLowerInvariant()
if ($actualSha256 -ne $ExpectedSha256.ToLowerInvariant()) {
    throw "SHA-256 mismatch: expected $ExpectedSha256, got $actualSha256"
}

Write-Output "Created $resolvedOutput"
Write-Output "SHA-256 verified: $actualSha256"
