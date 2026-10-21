param(
    [string]$RemoteRepo = '/root/autodl-tmp',
    [string]$Branch = 'master',
    [string]$Instance = '',
    [string[]]$RemoveOld = @(
        'sft_competitive_t2t_ultrachat_16k_coding_heavy.jsonl',
        'sft_easy_medium_competitive_clean_4k.jsonl'
    ),
    [switch]$SkipCacheCleanup,
    [switch]$ResumeOnly,
    [switch]$Execute
)

$ErrorActionPreference = 'Stop'
$relayProjectRoot = Split-Path -Parent $PSScriptRoot
$relayExecutable = Join-Path $relayProjectRoot 'data_relay_delivery.exe'
$relaySourceFile = Join-Path $relayProjectRoot 'dataset\pretrain_coder_12b.parquet'
$relayWorkingDir = Join-Path $relayProjectRoot 'data_relay'
foreach ($relayRequiredFile in @($relayExecutable, $relaySourceFile, (Join-Path $relayWorkingDir 'config.toml'))) {
    if (-not (Test-Path -LiteralPath $relayRequiredFile -PathType Leaf)) {
        throw "Required file missing: $relayRequiredFile"
    }
}

$relayArguments = @('--config', 'config.toml', 'deliver', '--source', $relaySourceFile,
    '--repo', $RemoteRepo, '--branch', $Branch, '--init-repo')
if ($ResumeOnly) {
    $RemoveOld = @()
    $SkipCacheCleanup = $true
    $relayArguments += '--skip-git-update'
}
foreach ($relayOldFile in $RemoveOld) {
    $relayArguments += @('--remove-old', $relayOldFile)
}
if ($Instance) { $relayArguments += @('--instance', $Instance) }
if (-not $SkipCacheCleanup) { $relayArguments += '--clean-cache' }
if ($Execute) { $relayArguments += '--execute' }

$relayLogDir = Join-Path $relayWorkingDir 'relay_data'
New-Item -ItemType Directory -Path $relayLogDir -Force | Out-Null
$relayStatePath = Join-Path $relayLogDir 'delivery-active.json'
$relayState = $null
$relayLock = $null
$relayExitCode = 1
function Save-RelayState {
    $relayStateTemp = $relayStatePath + '.tmp'
    [System.IO.File]::WriteAllText($relayStateTemp, ($relayState | ConvertTo-Json -Depth 5), [System.Text.UTF8Encoding]::new($false))
    if ([System.IO.File]::Exists($relayStatePath)) {
        # Windows PowerShell converts $null string arguments to an empty path.
        [System.IO.File]::Replace($relayStateTemp, $relayStatePath, ($relayStatePath + '.bak'))
    } else {
        [System.IO.File]::Move($relayStateTemp, $relayStatePath)
    }
}
try {
    # A filesystem lock also prevents duplicate workers in different Windows sessions.
    $relayLock = [System.IO.File]::Open((Join-Path $relayLogDir 'delivery.lock'), [System.IO.FileMode]::OpenOrCreate, [System.IO.FileAccess]::ReadWrite, [System.IO.FileShare]::None)
    $relayLogPath = Join-Path $relayLogDir ('delivery-' + (Get-Date -Format 'yyyyMMdd-HHmmss') + '.log')
    $relayState = [ordered]@{
        worker_pid = $PID; status = 'running'; log = $relayLogPath
        source = $relaySourceFile; remote_repo = $RemoteRepo; branch = $Branch
        resume_only = [bool]$ResumeOnly; started_at = (Get-Date -Format o)
        finished_at = $null; exit_code = $null; error = $null
    }
    Save-RelayState
    if ($Execute -and -not $ResumeOnly) {
        # The server cannot reach GitHub; fetch locally and deliver real Git objects.
        & git -C $relayProjectRoot fetch origin $Branch
        if ($LASTEXITCODE -ne 0) { throw 'Local git fetch failed.' }
        $relayBundlePath = Join-Path $relayLogDir ('instinct-' + ($Branch -replace '[^a-zA-Z0-9_-]', '_') + '.bundle')
        & git -C $relayProjectRoot bundle create $relayBundlePath "refs/remotes/origin/$Branch"
        if ($LASTEXITCODE -ne 0) { throw 'Git bundle creation failed.' }
        $relayArguments += @('--git-bundle', $relayBundlePath)
    }
    Write-Host "Target: $RemoteRepo/dataset; branch: $Branch; execute: $Execute; resume only: $ResumeOnly"
    Write-Host "Log: $relayLogPath"
    Push-Location -LiteralPath $relayWorkingDir
    try {
        $ErrorActionPreference = 'Continue'
        & $relayExecutable @relayArguments 2>&1 | Tee-Object -FilePath $relayLogPath
        $relayExitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = 'Stop'
        Pop-Location
    }
    if ($relayExitCode -ne 0) { throw "Data relay exited with code $relayExitCode. See $relayLogPath" }
    $relayState.status = 'completed'
} catch {
    $relayExitCode = 1
    if ($null -ne $relayState) {
        $relayState.status = 'failed'
        $relayState.error = $_.Exception.Message
    }
    [Console]::Error.WriteLine($_.Exception.Message)
} finally {
    if ($null -ne $relayState) {
        $relayState.exit_code = $relayExitCode
        $relayState.finished_at = Get-Date -Format o
        Save-RelayState
    }
    if ($null -ne $relayLock) { $relayLock.Dispose() }
}
exit $relayExitCode
