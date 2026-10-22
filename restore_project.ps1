param(
  [string]$PartsDirectory = $PSScriptRoot,
  [string]$OutputDirectory = (Join-Path $PSScriptRoot ('restored-' + (Get-Date -Format 'yyyyMMdd-HHmmss')))
)
$ErrorActionPreference = 'Stop'
$manifestPath = Join-Path $PartsDirectory 'parts.json'
if (-not (Test-Path -LiteralPath $manifestPath)) {
  $manifestPath = Join-Path $PSScriptRoot 'backup\parts.json'
}
if (-not (Test-Path -LiteralPath $manifestPath)) { throw 'parts.json was not found. Download it with every Release part.' }
$manifest = Get-Content -LiteralPath $manifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
if (Test-Path -LiteralPath $OutputDirectory) { throw 'Output directory already exists. Choose a new OutputDirectory to avoid overwriting files.' }
foreach ($part in $manifest.parts) {
  $path = Join-Path $PartsDirectory $part.name
  if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw ('Missing archive part: ' + $part.name) }
  if ((Get-Item -LiteralPath $path).Length -ne $part.bytes) { throw ('Wrong part size: ' + $part.name) }
  if ((Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant() -ne $part.sha256) { throw ('Part SHA-256 mismatch: ' + $part.name) }
  Write-Host ('Verified: ' + $part.name)
}
New-Item -ItemType Directory -Path $OutputDirectory | Out-Null
$zipPath = Join-Path $OutputDirectory $manifest.archive_name
$destination = [System.IO.File]::Open($zipPath, [System.IO.FileMode]::CreateNew)
try {
  foreach ($part in $manifest.parts) {
    $source = [System.IO.File]::OpenRead((Join-Path $PartsDirectory $part.name))
    try { $source.CopyTo($destination) } finally { $source.Dispose() }
  }
} finally { $destination.Dispose() }
if ((Get-FileHash -LiteralPath $zipPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $manifest.archive_sha256) { throw 'Combined ZIP SHA-256 mismatch.' }
Expand-Archive -LiteralPath $zipPath -DestinationPath (Join-Path $OutputDirectory 'files')
Write-Host ('Restore complete: ' + (Join-Path $OutputDirectory 'files'))
Write-Host 'The combined ZIP has been retained so this script does not delete any file.'
