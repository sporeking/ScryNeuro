[CmdletBinding()]
param(
    [string]$Python,
    [ValidateSet("debug", "release")]
    [string]$Profile = "release",
    [switch]$NoCopy
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Resolve-Executable {
    param(
        [string]$ExplicitPath,
        [string]$EnvironmentName,
        [string[]]$CommandNames
    )

    $candidate = $ExplicitPath
    if (-not $candidate -and $EnvironmentName) {
        $candidate = [Environment]::GetEnvironmentVariable($EnvironmentName)
    }
    if ($candidate) {
        $resolved = Resolve-Path -LiteralPath $candidate -ErrorAction Stop
        return $resolved.Path
    }

    foreach ($name in $CommandNames) {
        $command = Get-Command $name -CommandType Application -ErrorAction SilentlyContinue
        if ($command) {
            return $command.Source
        }
    }

    throw "Could not find an executable. Pass an explicit path or configure $EnvironmentName."
}

if ([Environment]::OSVersion.Platform -ne [PlatformID]::Win32NT) {
    throw "build_windows.ps1 only supports Windows."
}

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonExe = Resolve-Executable -ExplicitPath $Python -EnvironmentName "PYO3_PYTHON" -CommandNames @("python.exe", "python3.exe")
$cargoExe = Resolve-Executable -EnvironmentName "CARGO" -CommandNames @("cargo.exe")

$pythonInspector = @'
import json, os, platform, struct, sys
major, minor = sys.version_info[:2]
root = sys.base_prefix
print(json.dumps({
    "executable": sys.executable,
    "root": root,
    "prefix": sys.prefix,
    "version": platform.python_version(),
    "major": major,
    "minor": minor,
    "bits": struct.calcsize("P") * 8,
    "dll": os.path.join(root, f"python{major}{minor}.dll"),
    "import_lib": os.path.join(root, "libs", f"python{major}{minor}.lib"),
}))
'@
$pythonInfoJson = $pythonInspector | & $pythonExe -
if ($LASTEXITCODE -ne 0) {
    throw "Python inspection failed: $pythonExe"
}
$pythonInfo = $pythonInfoJson | ConvertFrom-Json

if ($pythonInfo.bits -ne 64) {
    throw "ScryNeuro for Windows currently requires 64-bit Python; found $($pythonInfo.bits)-bit."
}
if ($pythonInfo.major -ne 3 -or $pythonInfo.minor -lt 10 -or $pythonInfo.minor -gt 13) {
    throw "ScryNeuro currently supports Python 3.10 through 3.13; found $($pythonInfo.version)."
}
if (-not (Test-Path -LiteralPath $pythonInfo.dll -PathType Leaf)) {
    throw "Python runtime DLL not found: $($pythonInfo.dll)"
}
if (-not (Test-Path -LiteralPath $pythonInfo.import_lib -PathType Leaf)) {
    throw "Python import library not found: $($pythonInfo.import_lib)"
}

$previousPython = $env:PYO3_PYTHON
$previousPath = $env:PATH
try {
    $env:PYO3_PYTHON = $pythonExe
    $env:PATH = "$($pythonInfo.root);$env:PATH"

    Write-Host "Building ScryNeuro for Windows"
    Write-Host "  Python : $($pythonInfo.version) x64 ($pythonExe)"
    Write-Host "  Cargo  : $cargoExe"
    Write-Host "  Profile: $Profile"

    $cargoArgs = @("build", "--locked")
    if ($Profile -eq "release") {
        $cargoArgs += "--release"
    }

    Push-Location $projectRoot
    try {
        & $cargoExe @cargoArgs
        if ($LASTEXITCODE -ne 0) {
            throw "cargo build failed with exit code $LASTEXITCODE"
        }

        $targetRoot = [Environment]::GetEnvironmentVariable("CARGO_TARGET_DIR")
        if (-not $targetRoot) {
            $targetRoot = Join-Path $projectRoot "target"
        } elseif (-not [IO.Path]::IsPathRooted($targetRoot)) {
            $targetRoot = Join-Path $projectRoot $targetRoot
        }
        $targetTriple = [Environment]::GetEnvironmentVariable("CARGO_BUILD_TARGET")
        if ($targetTriple) {
            $targetRoot = Join-Path $targetRoot $targetTriple
        }
        $builtDll = Join-Path $targetRoot "$Profile\scryneuro.dll"
        if (-not (Test-Path -LiteralPath $builtDll -PathType Leaf)) {
            throw "Build completed but the expected DLL was not produced: $builtDll"
        }

        if (-not $NoCopy) {
            $destination = Join-Path $projectRoot "scryneuro.dll"
            Copy-Item -LiteralPath $builtDll -Destination $destination -Force
            Write-Host "Copied: $destination"
        }

        Write-Host "Build succeeded: $builtDll"
    } finally {
        Pop-Location
    }
} finally {
    $env:PYO3_PYTHON = $previousPython
    $env:PATH = $previousPath
}
