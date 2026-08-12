[CmdletBinding()]
param(
    [string]$Python,
    [string]$Scryer,
    [switch]$SkipBuild,
    [switch]$IncludeExistingSuites
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
        return (Resolve-Path -LiteralPath $candidate -ErrorAction Stop).Path
    }

    foreach ($name in $CommandNames) {
        $command = Get-Command $name -CommandType Application -ErrorAction SilentlyContinue
        if ($command) {
            return $command.Source
        }
    }

    throw "Could not find an executable. Pass an explicit path or configure $EnvironmentName."
}

function Invoke-ScryerTest {
    param(
        [string]$TestFile,
        [string]$ExpectedMarker
    )

    Write-Host "Running: $TestFile"
    $savedErrorActionPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        $output = & $script:ScryerExe -f $TestFile 2>&1
        $exitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $savedErrorActionPreference
    }
    $output | ForEach-Object { Write-Host $_ }
    if ($exitCode -ne 0) {
        throw "Scryer test failed with exit code ${exitCode}: $TestFile"
    }
    if ($ExpectedMarker -and -not (($output | Out-String) -match [regex]::Escape($ExpectedMarker))) {
        throw "Scryer test did not emit its success marker '$ExpectedMarker': $TestFile"
    }
    if (($output | Out-String) -cmatch "\bFAIL\b") {
        throw "Scryer test reported a failure despite returning exit code zero: $TestFile"
    }
}

if ([Environment]::OSVersion.Platform -ne [PlatformID]::Win32NT) {
    throw "test_windows.ps1 only supports Windows."
}

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonExe = Resolve-Executable -ExplicitPath $Python -EnvironmentName "PYO3_PYTHON" -CommandNames @("python.exe", "python3.exe")
$script:ScryerExe = Resolve-Executable -ExplicitPath $Scryer -EnvironmentName "SCRYER_PROLOG" -CommandNames @("scryer-prolog.exe")

$pythonInspector = @'
import json, sys, sysconfig
print(json.dumps({
    "base_prefix": sys.base_prefix,
    "prefix": sys.prefix,
    "purelib": sysconfig.get_path("purelib"),
}))
'@
$pythonInfoJson = $pythonInspector | & $pythonExe -
if ($LASTEXITCODE -ne 0 -or -not $pythonInfoJson) {
    throw "Could not inspect the selected Python runtime."
}
$pythonInfo = $pythonInfoJson | ConvertFrom-Json
$pythonRoot = $pythonInfo.base_prefix
$pythonPrefix = $pythonInfo.prefix
$pythonSitePackages = $pythonInfo.purelib

if (-not $SkipBuild) {
    & (Join-Path $projectRoot "build_windows.ps1") -Python $pythonExe
    if ($LASTEXITCODE -ne 0) {
        throw "Windows build failed."
    }
}

$dll = Join-Path $projectRoot "scryneuro.dll"
if (-not (Test-Path -LiteralPath $dll -PathType Leaf)) {
    throw "ScryNeuro DLL not found: $dll"
}

$previousPath = $env:PATH
$previousHome = $env:SCRYNEURO_HOME
$previousPythonHome = $env:PYTHONHOME
$previousPythonPath = $env:PYTHONPATH
$nonAsciiSuffix = ([char]0x4E2D).ToString() + ([char]0x6587).ToString()
$tempRoot = Join-Path ([IO.Path]::GetTempPath()) ("ScryNeuro Windows path test " + [Guid]::NewGuid().ToString("N") + " " + $nonAsciiSuffix)
$fixtureTest = $null

try {
    $env:PATH = "$pythonRoot;$projectRoot;$env:PATH"
    $env:PYTHONHOME = $pythonRoot
    if ($pythonPrefix -ne $pythonRoot -and $pythonSitePackages) {
        if ($env:PYTHONPATH) {
            $env:PYTHONPATH = "$pythonSitePackages;$env:PYTHONPATH"
        } else {
            $env:PYTHONPATH = $pythonSitePackages
        }
    }
    $env:SCRYNEURO_HOME = $projectRoot

    $version = & $script:ScryerExe --version
    if ($LASTEXITCODE -ne 0) {
        throw "Scryer Prolog did not start: $script:ScryerExe"
    }
    Write-Host "Scryer : $version"
    Write-Host "Python : $(& $pythonExe --version)"

    Push-Location $projectRoot
    try {
        Invoke-ScryerTest -TestFile "test/test_windows_ffi_abi.pl" -ExpectedMarker "WINDOWS FFI ABI PASSED"
        Invoke-ScryerTest -TestFile "test/test_windows.pl" -ExpectedMarker "WINDOWS TESTS PASSED"
        if ($IncludeExistingSuites) {
            Invoke-ScryerTest -TestFile "test/test_smoke.pl" -ExpectedMarker "All tests passed!"
            Invoke-ScryerTest -TestFile "test/test_pi.pl" -ExpectedMarker "done"
            Invoke-ScryerTest -TestFile "test/test_comprehensive.pl" -ExpectedMarker "=== ALL 26 TESTS PASSED ==="
            Invoke-ScryerTest -TestFile "test/test_minimal_api.pl" -ExpectedMarker "[=,=,=, ,D,O,N,E, ,=,=,=]"
            Invoke-ScryerTest -TestFile "test/test_prolog_api.pl" -ExpectedMarker "=== ALL 32 PROLOG API TESTS PASSED ==="
        }
    } finally {
        Pop-Location
    }

    New-Item -ItemType Directory -Path $tempRoot | Out-Null
    Copy-Item -LiteralPath $dll -Destination (Join-Path $tempRoot "scryneuro.dll")
    $fixtureDir = Join-Path $tempRoot "python"
    New-Item -ItemType Directory -Path $fixtureDir | Out-Null
    Set-Content -LiteralPath (Join-Path $fixtureDir "windows_path_fixture.py") -Encoding Ascii -Value "VALUE = 314159"
    $fixtureTest = Join-Path $projectRoot (".scryneuro-home-fixture-" + [Guid]::NewGuid().ToString("N") + ".pl")
    $fixtureSource = @"
:- use_module('prolog/scryer_py').
:- use_module(library(format)).

run_test :-
    py_init,
    py_import("windows_path_fixture", Module),
    py_getattr(Module, "VALUE", ValueHandle),
    py_to_int(ValueHandle, Value),
    Value =:= 314159,
    py_free(ValueHandle),
    py_free(Module),
    py_finalize,
    format("WINDOWS HOME MODULE PASSED~n", []).

main :-
    catch(run_test, Error, (format("WINDOWS HOME MODULE FAILURE: ~q~n", [Error]), halt(1))),
    halt.

:- initialization(main).
"@
    $utf8NoBom = New-Object Text.UTF8Encoding($false)
    [IO.File]::WriteAllText($fixtureTest, $fixtureSource, $utf8NoBom)
    $env:SCRYNEURO_HOME = $tempRoot

    Push-Location $projectRoot
    try {
        Invoke-ScryerTest -TestFile "test/test_windows.pl" -ExpectedMarker "WINDOWS TESTS PASSED"
        Write-Host "Running: SCRYNEURO_HOME/python import fixture"
        $savedErrorActionPreference = $ErrorActionPreference
        try {
            $ErrorActionPreference = "Continue"
            $fixtureOutput = & $script:ScryerExe -f $fixtureTest 2>&1
            $fixtureExitCode = $LASTEXITCODE
        } finally {
            $ErrorActionPreference = $savedErrorActionPreference
        }
        $fixtureOutput | ForEach-Object { Write-Host $_ }
        $fixtureText = $fixtureOutput | Out-String
        if ($fixtureExitCode -ne 0 -or $fixtureText -cmatch "\bFAIL(?:URE)?\b" -or -not ($fixtureText -match "(?m)^WINDOWS HOME MODULE PASSED\r?$")) {
            throw "SCRYNEURO_HOME Python module import test failed."
        }
    } finally {
        Pop-Location
    }

    Write-Host "All requested Windows tests passed, including a path with spaces and non-ASCII characters."
} finally {
    $env:PATH = $previousPath
    $env:SCRYNEURO_HOME = $previousHome
    $env:PYTHONHOME = $previousPythonHome
    $env:PYTHONPATH = $previousPythonPath
    if (Test-Path -LiteralPath $tempRoot) {
        Remove-Item -LiteralPath $tempRoot -Recurse -Force
    }
    if ($fixtureTest -and (Test-Path -LiteralPath $fixtureTest)) {
        Remove-Item -LiteralPath $fixtureTest -Force
    }
}
