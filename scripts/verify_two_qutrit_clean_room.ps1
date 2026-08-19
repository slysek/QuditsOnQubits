param(
    [string]$Python = "python",
    [int]$Shots = 2048,
    [switch]$ReuseSystemPackages
)

$ErrorActionPreference = "Stop"

if ($Shots -lt 256) {
    throw "Shots must be at least 256 for Bell-value verification."
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$tempRoot = Join-Path ([IO.Path]::GetTempPath()) ("qoq-clean-room-" + [guid]::NewGuid().ToString("N"))
$sourceDir = Join-Path $tempRoot "source"
$wheelDir = Join-Path $tempRoot "wheel"
$venvDir = Join-Path $tempRoot "venv"
$outputRoot = Join-Path $tempRoot "artifacts"

New-Item -ItemType Directory -Path $wheelDir -Force | Out-Null
New-Item -ItemType Directory -Path $sourceDir -Force | Out-Null
Copy-Item -LiteralPath (Join-Path $repoRoot "pyproject.toml") -Destination $sourceDir
Copy-Item -LiteralPath (Join-Path $repoRoot "README.md") -Destination $sourceDir
Copy-Item -LiteralPath (Join-Path $repoRoot "src") -Destination $sourceDir -Recurse


try {
    & $Python -m pip wheel $sourceDir --no-deps --no-build-isolation --wheel-dir $wheelDir
    if ($LASTEXITCODE -ne 0) {
        throw "Wheel build failed."
    }

    $venvArguments = @("-m", "venv")
    if ($ReuseSystemPackages) {
        $venvArguments += "--system-site-packages"
    }
    $venvArguments += $venvDir
    & $Python @venvArguments
    if ($LASTEXITCODE -ne 0) {
        throw "Virtual environment creation failed."
    }

    $isWindows = [Environment]::OSVersion.Platform -eq [PlatformID]::Win32NT
    if ($isWindows) {
        $venvPython = Join-Path $venvDir "Scripts/python.exe"
        $command = Join-Path $venvDir "Scripts/qoq-two-qutrit-bell.exe"
    } else {
        $venvPython = Join-Path $venvDir "bin/python"
        $command = Join-Path $venvDir "bin/qoq-two-qutrit-bell"
    }

    $wheel = Get-ChildItem -LiteralPath $wheelDir -Filter "qudits_on_qubits-*.whl" | Select-Object -First 1
    if ($null -eq $wheel) {
        throw "Built wheel was not found."
    }
    $installArguments = @("-m", "pip", "install")
    if ($ReuseSystemPackages) {
        $installArguments += "--no-deps"
    }
    $installArguments += $wheel.FullName
    & $venvPython @installArguments
    if ($LASTEXITCODE -ne 0) {
        throw "Wheel installation failed."
    }

    $output = & $command --shots $Shots --seed 42 --output-root $outputRoot
    if ($LASTEXITCODE -ne 0) {
        throw "Vertical-slice command failed."
    }
    $output | Write-Output

    $manifestLine = $output | Where-Object { $_ -like "manifest=*" } | Select-Object -First 1
    if ($null -eq $manifestLine) {
        throw "Command did not print a manifest path."
    }
    $manifestPath = $manifestLine.Substring("manifest=".Length)
    $manifest = Get-Content -Raw -LiteralPath $manifestPath | ConvertFrom-Json
    if ($manifest.schema_version -ne "run-manifest-v1") {
        throw "Unexpected manifest schema version."
    }
    if ($manifest.status -ne "completed") {
        throw "Run did not complete."
    }
    if ($manifest.result.circuit_count -ne 9) {
        throw "Expected 9 Bell measurement circuits."
    }
    if ([double]$manifest.result.leakage_rate -ne 0.0) {
        throw "Ideal reference run reported leakage."
    }
    $bellValue = [double]$manifest.result.bell_unconditional.real
    $bellTolerance = 0.15 * [Math]::Sqrt(2048.0 / $Shots)
    if ([Math]::Abs($bellValue - 6.0) -gt $bellTolerance) {
        throw "Bell value is outside expected tolerance: $bellValue"
    }
    if ($manifest.artifacts.Count -ne 7) {
        throw "Expected 7 integrity-linked artifacts."
    }
    Write-Output "clean_room_vertical_slice=passed"
} finally {
    $resolvedTemp = [IO.Path]::GetFullPath($tempRoot)
    $systemTemp = [IO.Path]::GetFullPath([IO.Path]::GetTempPath())
    if ($resolvedTemp.StartsWith($systemTemp, [StringComparison]::OrdinalIgnoreCase) -and (Test-Path -LiteralPath $resolvedTemp)) {
        Remove-Item -LiteralPath $resolvedTemp -Recurse -Force
    }
}
