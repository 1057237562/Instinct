param(
    [string]$RemoteRepo = '/root/autodl-tmp',
    [string]$Branch = 'master',
    [string]$Instance = '',
    [switch]$ResumeOnly,
    [switch]$SkipCacheCleanup
)

$ErrorActionPreference = 'Stop'
$relayProjectRoot = Split-Path -Parent $PSScriptRoot
$relayWorker = Join-Path $PSScriptRoot 'upload_pretrain_coder_12b.ps1'
$relayStatePath = Join-Path $relayProjectRoot 'data_relay\relay_data\delivery-active.json'
if (Test-Path -LiteralPath $relayStatePath) {
    $relayPreviousState = Get-Content -LiteralPath $relayStatePath -Raw | ConvertFrom-Json
    if ($relayPreviousState.status -eq 'running') {
        $relayPreviousWorker = Get-CimInstance Win32_Process -Filter "ProcessId = $($relayPreviousState.worker_pid)"
        if ($relayPreviousWorker -and $relayPreviousWorker.CommandLine -match [regex]::Escape($relayWorker)) {
            Write-Host "Upload is already running: PID $($relayPreviousState.worker_pid)"
            Write-Host "Log: $($relayPreviousState.log)"
            exit 0
        }
    }
    if ($relayPreviousState.remote_repo -eq $RemoteRepo -and $relayPreviousState.branch -eq $Branch -and -not $Instance) {
        if ($relayPreviousState.status -eq 'completed' -and -not $ResumeOnly) {
            Write-Host "Delivery is already completed. Log: $($relayPreviousState.log)"
            exit 0
        }
        # A stale running state or a failed worker resumes without repeating cleanup/Git.
        $ResumeOnly = $true
    }
}
$relayWorkerArgs = @('-NoProfile', '-WindowStyle', 'Hidden', '-ExecutionPolicy', 'Bypass',
    '-File', $relayWorker, '-Execute', '-RemoteRepo', $RemoteRepo, '-Branch', $Branch)
if ($Instance) { $relayWorkerArgs += @('-Instance', $Instance) }
if ($ResumeOnly) { $relayWorkerArgs += '-ResumeOnly' }
if ($SkipCacheCleanup) { $relayWorkerArgs += '-SkipCacheCleanup' }
function Quote-RelayArg([string]$Value) {
    if ($Value.Contains('"') -or $Value.Contains("`r") -or $Value.Contains("`n")) {
        throw 'Quotes and newlines are not supported in launcher parameters.'
    }
    return '"' + $Value + '"'
}
$relayPowerShell = Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe'
$relayCommand = (Quote-RelayArg $relayPowerShell) + ' ' + (($relayWorkerArgs | ForEach-Object { Quote-RelayArg $_ }) -join ' ')
# WMI owns the worker, so it does not inherit the Codex terminal/process lifetime.
$relayStartup = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ ShowWindow = [uint16]0 }
$relayLaunch = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
    CommandLine = $relayCommand; CurrentDirectory = $relayProjectRoot
    ProcessStartupInformation = $relayStartup
}
if ($relayLaunch.ReturnValue -ne 0) {
    throw "Independent worker launch failed (WMI code $($relayLaunch.ReturnValue))."
}
Write-Host "Independent upload worker started: PID $($relayLaunch.ProcessId)"
Write-Host ('State: ' + $relayStatePath)
Write-Host 'The worker keeps running when this window or Codex closes.'
