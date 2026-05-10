param(
    [string]$Ref = "41d924371612b692c0fd1e4d9d94c3dfb3c02cb3"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$target = Join-Path $root "third_party\LTX-2"
if (-not (Test-Path $target)) {
    git clone https://github.com/Lightricks/LTX-2.git $target
}
git -C $target fetch origin
git -C $target checkout $Ref
Write-Host "LTX-2 checked out at $Ref"

