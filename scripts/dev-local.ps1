param(
    [Parameter(Mandatory=$true)]
    [ValidateSet("gateway", "gateway-static", "text", "ltx", "frontend")]
    [string]$Service
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:UV_CACHE_DIR = Join-Path $root ".uv-cache"
New-Item -ItemType Directory -Force -Path $env:UV_CACHE_DIR | Out-Null

switch ($Service) {
    "gateway" {
        cargo run --manifest-path gateway-rs\Cargo.toml
    }
    "gateway-static" {
        npm.cmd --prefix frontend run build
        $env:FRONTEND_DIR = Join-Path $root "frontend\dist"
        cargo run --manifest-path gateway-rs\Cargo.toml
    }
    "text" {
        Push-Location (Join-Path $root "services\text-worker")
        try {
            uv run uvicorn app.main:app --host 127.0.0.1 --port 8101
        } finally {
            Pop-Location
        }
    }
    "ltx" {
        Push-Location (Join-Path $root "services\ltx-worker")
        try {
            uv run uvicorn app.main:app --host 127.0.0.1 --port 8102
        } finally {
            Pop-Location
        }
    }
    "frontend" {
        Push-Location (Join-Path $root "frontend")
        try {
            npm.cmd run dev
        } finally {
            Pop-Location
        }
    }
}
