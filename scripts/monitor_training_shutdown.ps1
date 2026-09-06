[CmdletBinding()]
param(
    [ValidateRange(0, 86400)]
    [int]$ShutdownDelaySeconds = 60,

    [switch]$Detach,

    [switch]$NoShutdown
)

$ErrorActionPreference = 'Stop'

# A detached monitor is useful when the current terminal should remain free.
if ($Detach) {
    $repoRoot = Split-Path -Parent $PSScriptRoot
    $logDir = Join-Path $repoRoot 'checkpoints'
    New-Item -ItemType Directory -Path $logDir -Force | Out-Null

    $stdoutLog = Join-Path $logDir 'training_shutdown_monitor.log'
    $stderrLog = Join-Path $logDir 'training_shutdown_monitor.error.log'
    $arguments = @(
        '-NoProfile',
        '-ExecutionPolicy', 'Bypass',
        '-File', ('"' + $PSCommandPath + '"'),
        '-ShutdownDelaySeconds', [string]$ShutdownDelaySeconds
    )
    if ($NoShutdown) {
        $arguments += '-NoShutdown'
    }

    $monitor = Start-Process -FilePath 'powershell.exe' `
        -ArgumentList $arguments `
        -WindowStyle Hidden `
        -RedirectStandardOutput $stdoutLog `
        -RedirectStandardError $stderrLog `
        -PassThru

    Write-Host "Detached monitor started (PID $($monitor.Id))."
    Write-Host "Log: $stdoutLog"
    Write-Host "Error log: $stderrLog"
    exit 0
}

$trainingPattern = '(?i)trainer[\\/]+train_(pretrain|full_sft|lora|dpo|ppo|grpo|agent|distillation)\.py'
$allProcesses = @(Get-CimInstance Win32_Process)
$candidates = @(
    $allProcesses | Where-Object {
        $_.ProcessId -ne $PID -and
        $_.CommandLine -and
        $_.CommandLine -match $trainingPattern
    }
)

if ($candidates.Count -eq 0) {
    Write-Error 'No running Instinct training process was found.'
    exit 2
}

$allByPid = @{}
foreach ($processInfo in $allProcesses) {
    $allByPid[[int]$processInfo.ProcessId] = $processInfo
}

$candidateByPid = @{}
foreach ($candidate in $candidates) {
    $candidateByPid[[int]$candidate.ProcessId] = $candidate
}

function Test-HasTrainingAncestor {
    param([Parameter(Mandatory)]$ProcessInfo)

    $parentId = [int]$ProcessInfo.ParentProcessId
    $visited = @{}
    while ($parentId -gt 0 -and $allByPid.ContainsKey($parentId)) {
        if ($visited.ContainsKey($parentId)) {
            break
        }
        $visited[$parentId] = $true
        if ($candidateByPid.ContainsKey($parentId)) {
            return $true
        }
        $parentId = [int]$allByPid[$parentId].ParentProcessId
    }
    return $false
}

# For torchrun, both the launcher and workers may contain the trainer path in
# their command line. Monitor only the highest matching process in each tree.
$roots = @(
    $candidates | Where-Object { -not (Test-HasTrainingAncestor $_) }
)

if ($roots.Count -ne 1) {
    Write-Host "Found $($roots.Count) independent training process trees."
    Write-Host 'Automatic shutdown is disabled because choosing one would be unsafe.'
    $roots |
        Select-Object ProcessId, ParentProcessId, CommandLine |
        Format-Table -AutoSize -Wrap
    exit 3
}

$trainingRoot = $roots[0]
$trainingProcessId = [int]$trainingRoot.ProcessId
Write-Host "Monitoring training PID $trainingProcessId"
Write-Host $trainingRoot.CommandLine

try {
    $trainingProcess = [System.Diagnostics.Process]::GetProcessById($trainingProcessId)
    $trainingProcess.WaitForExit()
    $exitCode = $trainingProcess.ExitCode
}
catch {
    Write-Error "Could not monitor training PID ${trainingProcessId}: $($_.Exception.Message)"
    exit 4
}

Write-Host "Training process exited with code $exitCode."

if ($exitCode -eq 42) {
    Write-Host 'Training was paused. The computer will remain on.'
    exit 0
}

if ($exitCode -ne 0) {
    Write-Host 'Training failed or was terminated. The computer will remain on.'
    exit $exitCode
}

if ($NoShutdown) {
    Write-Host 'Training completed normally. NoShutdown is set, so shutdown was skipped.'
    exit 0
}

$shutdownExe = Join-Path $env:SystemRoot 'System32\shutdown.exe'
$comment = "Instinct training completed normally. Shutdown in $ShutdownDelaySeconds seconds."
& $shutdownExe /s /t $ShutdownDelaySeconds /c $comment
if ($LASTEXITCODE -ne 0) {
    Write-Error "shutdown.exe failed with exit code $LASTEXITCODE."
    exit $LASTEXITCODE
}

Write-Host "Training completed normally. Shutdown scheduled in $ShutdownDelaySeconds seconds."
Write-Host 'Run shutdown.exe /a to cancel it.'
