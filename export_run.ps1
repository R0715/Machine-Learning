[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Run,
    [string]$Dataset,
    [string]$Prices,
    [string]$CpuPython = 'D:/Anaconda3/python.exe',
    [string]$GpuPython = 'D:/Anaconda3/envs/yolo_env/python.exe'
)

$ErrorActionPreference = 'Stop'
$Run = (Resolve-Path -LiteralPath $Run).Path
if (-not $Dataset) {
    $bundled = @(Get-ChildItem -LiteralPath (Join-Path $PSScriptRoot 'data') -Filter '*.zip' -File)
    if ($bundled.Count -ne 1) { throw 'Pass -Dataset with the revised data ZIP path.' }
    $Dataset = $bundled[0].FullName
}
if (-not $Prices) { $Prices = Join-Path $PSScriptRoot 'data/clean_prices.csv' }
$Dataset = (Resolve-Path -LiteralPath $Dataset).Path
$Prices = (Resolve-Path -LiteralPath $Prices).Path
$entry = Join-Path $PSScriptRoot 'export_predictions.py'

& $CpuPython -B -X utf8 -u $entry infer --component (Join-Path $Run 'cpu') --dataset $Dataset --device cpu
if ($LASTEXITCODE -ne 0) { throw "CPU checkpoint inference exited with code $LASTEXITCODE." }
& $GpuPython -B -X utf8 -u $entry infer --component (Join-Path $Run 'gpu') --dataset $Dataset --device cuda
if ($LASTEXITCODE -ne 0) { throw "GPU checkpoint inference exited with code $LASTEXITCODE." }
& $CpuPython -B -X utf8 -u $entry combine --run $Run --dataset $Dataset --prices $Prices
if ($LASTEXITCODE -ne 0) { throw "Ten-column export exited with code $LASTEXITCODE." }
Write-Host "Export complete: $(Join-Path $Run 'handoff')"
