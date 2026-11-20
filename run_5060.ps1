[CmdletBinding()]
param(
    [string]$Dataset,
    [string]$CpuPython = 'D:/Anaconda3/python.exe',
    [string]$GpuPython = 'D:/Anaconda3/envs/yolo_env/python.exe',
    [string]$Prices,
    [string]$OutputRoot,
    [ValidateSet('both', 'close', 'trade')][string]$Task = 'both',
    [ValidateSet(32, 64)][int]$Lookback = 64,
    [ValidateRange(1, 10000)][int]$Epochs = 20,
    [ValidateRange(1, 10000)][int]$TqnetEpochs = 30,
    [ValidateRange(1, 4096)][int]$BatchSize = 64,
    [int]$Seed = 42,
    [switch]$PredictTest = $true
)

$ErrorActionPreference = 'Stop'
if (-not $Dataset) {
    $bundled = @(Get-ChildItem -LiteralPath (Join-Path $PSScriptRoot 'data') -Filter '*.zip' -File)
    if ($bundled.Count -ne 1) { throw 'Pass -Dataset with the path to the revised data ZIP.' }
    $Dataset = $bundled[0].FullName
}
$Dataset = (Resolve-Path -LiteralPath $Dataset).Path
foreach ($python in @($CpuPython, $GpuPython)) {
    if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
        throw "Python not found: $python. Set -CpuPython and -GpuPython."
    }
}
if (-not $OutputRoot) { $OutputRoot = Join-Path $PSScriptRoot 'runs' }
$runName = "$(Get-Date -Format yyyyMMdd_HHmmss)_$([guid]::NewGuid().ToString('N').Substring(0, 6))"
$runFolder = Join-Path ([IO.Path]::GetFullPath($OutputRoot)) $runName
New-Item -ItemType Directory -Path $runFolder | Out-Null
$cpuFolder = Join-Path $runFolder 'cpu'
$gpuFolder = Join-Path $runFolder 'gpu'
$entry = Join-Path $PSScriptRoot 'run_baselines.py'
$common = @(
    '--dataset', $Dataset, '--task', $Task, '--lookback', "$Lookback",
    '--epochs', "$Epochs", '--tqnet-epochs', "$TqnetEpochs",
    '--patchtst-lookback', '336', '--batch-size', "$BatchSize", '--seed', "$Seed"
)
if ($PredictTest) { $common += '--predict-test' }
if ($PredictTest -and $Task -eq 'both') {
    if (-not $Prices) { $Prices = Join-Path $PSScriptRoot 'data/clean_prices.csv' }
    $Prices = (Resolve-Path -LiteralPath $Prices).Path
}

& $CpuPython -B -X utf8 -u $entry @common --models ridge lightgbm --device cpu --output $cpuFolder 2>&1 |
    Tee-Object -FilePath (Join-Path $runFolder 'cpu.log')
if ($LASTEXITCODE -ne 0) { throw "CPU training exited with code $LASTEXITCODE." }

& $GpuPython -B -X utf8 -u $entry @common --models tqnet patchtst --device cuda --output $gpuFolder 2>&1 |
    Tee-Object -FilePath (Join-Path $runFolder 'gpu.log')
if ($LASTEXITCODE -ne 0) { throw "GPU training exited with code $LASTEXITCODE. CPU outputs are in $cpuFolder." }

Import-Csv -LiteralPath @(
    (Join-Path $cpuFolder 'validation_metrics.csv'),
    (Join-Path $gpuFolder 'validation_metrics.csv')
) | Export-Csv -LiteralPath (Join-Path $runFolder 'validation_metrics.csv') -NoTypeInformation -Encoding utf8
Write-Host "Combined metrics: $(Join-Path $runFolder 'validation_metrics.csv')"
if ($PredictTest -and $Task -eq 'both') {
    & $CpuPython -B -X utf8 (Join-Path $PSScriptRoot 'export_predictions.py') combine --run $runFolder --dataset $Dataset --prices $Prices
    if ($LASTEXITCODE -ne 0) { throw "Ten-column export exited with code $LASTEXITCODE." }
    & $CpuPython -B -X utf8 (Join-Path $PSScriptRoot 'evaluate_test.py') --run $runFolder --prices $Prices
    if ($LASTEXITCODE -ne 0) { throw "Test evaluation exited with code $LASTEXITCODE." }
    & $CpuPython -B -X utf8 (Join-Path $PSScriptRoot 'backtest_research.py') --run $runFolder --prices $Prices --config (Join-Path $PSScriptRoot 'config/strategy_frozen.json')
    if ($LASTEXITCODE -ne 0) { throw "Frozen strategy backtest exited with code $LASTEXITCODE." }
}
Set-Content -LiteralPath (Join-Path $PSScriptRoot 'latest_run.txt') -Value $runFolder -Encoding utf8
Write-Host "Completed run: $runFolder"
