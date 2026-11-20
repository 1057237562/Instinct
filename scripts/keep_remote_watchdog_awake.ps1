param(
    [Parameter(Mandatory=$true)][string]$AutomationFile,
    [Parameter(Mandatory=$true)][string]$WatchdogState
)
$ErrorActionPreference = 'Stop'
Add-Type -Namespace InstinctWatchdog -Name NativePower -MemberDefinition @'
[System.Runtime.InteropServices.DllImport("kernel32.dll")]
public static extern uint SetThreadExecutionState(uint flags);
'@
try {
    $result = [InstinctWatchdog.NativePower]::SetThreadExecutionState([uint32]2147483649)
    if ($result -eq 0) { throw 'Could not prevent system sleep during the active watchdog.' }
    Write-Output 'System sleep prevented while the 5090 watchdog remains active.'
    while ((Test-Path -LiteralPath $AutomationFile) -and
        (Select-String -LiteralPath $AutomationFile -Pattern '^status = "ACTIVE"$' -Quiet)) {
        if (Test-Path -LiteralPath $WatchdogState) {
            $state = Get-Content -LiteralPath $WatchdogState -Raw | ConvertFrom-Json
            if ($state.shutdown_scheduled) { break }
        }
        Start-Sleep -Seconds 30
    }
}
finally {
    [void][InstinctWatchdog.NativePower]::SetThreadExecutionState([uint32]2147483648)
}
