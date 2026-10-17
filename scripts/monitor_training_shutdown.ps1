[CmdletBinding()]
param(
    [ValidateRange(0, 86400)]
    [int]$ShutdownDelaySeconds = 60,

    # Explicit PID of the training/pipeline process to watch. The Config WebUI
    # arms the detached monitor this way right after spawning the process, so
    # shutdown still happens when no browser session is polling anymore.
    [ValidateRange(0, [int]::MaxValue)]
    [int]$ProcessId = 0,

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
    if ($ProcessId -gt 0) {
        $arguments += '-ProcessId'
        $arguments += [string]$ProcessId
    }
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

$trainingProcessId = 0
if ($ProcessId -gt 0) {
    # Explicit-PID mode: the WebUI spawned the process it wants monitored, so
    # no command-line pattern or process-tree disambiguation is needed.
    $trainingProcessId = [int]$ProcessId
    Write-Host "Monitoring explicit training PID $trainingProcessId"
}
else {
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
}

try {
    $trainingProcess = [System.Diagnostics.Process]::GetProcessById($trainingProcessId)
    # Open the process handle while the process is still alive. Without this,
    # .NET reads ExitCode back as empty after WaitForExit on a foreign PID and
    # the script would always treat a successful run as failed.
    $null = $trainingProcess.Handle
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

# The Config WebUI persists its panel state on every interaction. Read the
# freshest auto-shutdown choice from it so un-ticking the checkbox mid-run
# disarms this monitor even when the browser has since been closed.
$webuiStatePath = Join-Path (Split-Path -Parent $PSScriptRoot) 'trainer\webui_state.json'
if (Test-Path $webuiStatePath) {
    try {
        $webuiState = Get-Content $webuiStatePath -Raw -Encoding UTF8 | ConvertFrom-Json
        $shutdownFlag = $webuiState.shutdown_after_train
        if ($null -ne $shutdownFlag -and -not [bool]$shutdownFlag) {
            Write-Host 'Auto-shutdown is disabled in the Config WebUI state. The computer will remain on.'
            exit 0
        }
        $stateDelay = $webuiState.shutdown_delay_seconds
        if ($null -ne $stateDelay) {
            $ShutdownDelaySeconds = [Math]::Min(86400, [Math]::Max(10, [int]$stateDelay))
        }
    }
    catch {
        Write-Host "Could not parse webui_state.json ($($_.Exception.Message)); using the armed settings."
    }
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
