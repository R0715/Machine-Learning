[CmdletBinding()]
param(
    [string]$CpuPython = 'D:/Anaconda3/python.exe',
    [string]$GpuPython = 'D:/Anaconda3/envs/yolo_env/python.exe',
    [ValidateSet('cpu', 'cuda')][string]$SequenceDevice = 'cuda',
    [string]$Output
)
$ErrorActionPreference = 'Stop'
foreach ($python in @($CpuPython, $GpuPython)) {
    if (-not (Test-Path -LiteralPath $python -PathType Leaf)) { throw "Python not found: $python" }
}
if (-not $Output) {
    $Output = Join-Path $PSScriptRoot ("work/reproduction_" + (Get-Date -Format yyyyMMdd_HHmmss) + '_' + [guid]::NewGuid().ToString('N').Substring(0, 6))
}
$Output = [IO.Path]::GetFullPath($Output)
if (Test-Path -LiteralPath $Output) { throw "Use a new output directory: $Output" }
$entry = Join-Path $PSScriptRoot 'reproduce_final.py'
& $CpuPython -B -X utf8 $entry --stage cpu --output $Output
if ($LASTEXITCODE -ne 0) { throw 'CPU checkpoint verification failed.' }
& $GpuPython -B -X utf8 $entry --stage gpu --output $Output --device $SequenceDevice
if ($LASTEXITCODE -ne 0) { throw 'Sequence checkpoint verification failed.' }
& $CpuPython -B -X utf8 $entry --stage combine --output $Output
if ($LASTEXITCODE -ne 0) { throw 'Prediction export or frozen-strategy verification failed.' }
Write-Host "Completed reproduction: $Output"
