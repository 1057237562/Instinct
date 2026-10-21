$ErrorActionPreference = 'Stop'
$taskWorkspace = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path.TrimEnd('\')
$taskDataset = Join-Path $taskWorkspace 'dataset'
$taskTargets = [System.Collections.Generic.List[object]]::new()
$taskSeen = [System.Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
$taskRelease = Join-Path $taskDataset 'pretrain_ultradata_l3_cpt_12b_general10_identity_clean.parquet'
$taskAudit = Get-Content -LiteralPath ($taskRelease + '.verification.json') -Raw -Encoding UTF8 | ConvertFrom-Json
if ($taskAudit.passed -ne $true) { throw 'The final artifact audit must pass before cleanup.' }
$taskActive = @(Get-CimInstance Win32_Process | Where-Object {
    ($_.Name -eq 'python.exe' -or $_.Name -eq 'dataset_compiler.exe') -and
    $_.CommandLine -and $_.CommandLine.Contains($taskWorkspace)
})
if ($taskActive.Count) { throw 'Repository Python/compiler processes are still active; keep their caches.' }

function Add-TaskCacheFile([string]$Candidate, [string]$AllowedRoot, [string]$Reason) {
    if (-not (Test-Path -LiteralPath $Candidate -PathType Leaf)) { return }
    $taskResolved = (Resolve-Path -LiteralPath $Candidate).Path
    $taskBoundary = (Resolve-Path -LiteralPath $AllowedRoot).Path.TrimEnd('\') + '\'
    if (-not $taskResolved.StartsWith($taskBoundary, [StringComparison]::OrdinalIgnoreCase)) {
        throw "File escaped its reviewed cache root: $taskResolved"
    }
    $taskItem = Get-Item -LiteralPath $taskResolved
    if ($taskItem.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw "Skip linked cache: $taskResolved" }
    if ($taskSeen.Add($taskResolved)) {
        $taskTargets.Add([pscustomobject]@{path=$taskResolved;kind='file';bytes=$taskItem.Length;reason=$Reason})
    }
}

function Add-TaskCacheDirectory([string]$Candidate, [string]$Reason) {
    if (-not (Test-Path -LiteralPath $Candidate -PathType Container)) { return }
    $taskResolved = (Resolve-Path -LiteralPath $Candidate).Path.TrimEnd('\')
    # Verify the resolved absolute directory before ANY recursive operation.
    if (-not $taskResolved.StartsWith($taskWorkspace + '\', [StringComparison]::OrdinalIgnoreCase)) {
        throw "Directory escaped workspace: $taskResolved"
    }
    if ((Get-Item -LiteralPath $taskResolved).Attributes -band [IO.FileAttributes]::ReparsePoint) {
        throw "Skip linked directory: $taskResolved"
    }
    $taskLinks = @(Get-ChildItem -LiteralPath $taskResolved -Recurse -Force -Attributes ReparsePoint)
    if ($taskLinks.Count) { throw "Cache contains linked paths: $taskResolved" }
    $taskSize = (Get-ChildItem -LiteralPath $taskResolved -Recurse -File -Force | Measure-Object Length -Sum).Sum
    if ($taskSeen.Add($taskResolved)) {
        $taskTargets.Add([pscustomobject]@{path=$taskResolved;kind='directory';bytes=[long]$taskSize;reason=$Reason})
    }
}

# Only output shards named by the task's filtering reports. Preserve all
# candidates, decisions, audit samples, fingerprint indexes and reports.
$taskL3 = Join-Path $taskDataset 'ultradata_code_l3_12b\quality_clean_v1'
foreach ($taskReport in Get-ChildItem -LiteralPath $taskL3 -Filter 'UltraData*.report.json' -File) {
    $taskRecord = Get-Content -LiteralPath $taskReport.FullName -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($taskRecord.output) {
        $taskName = [IO.Path]::GetFileName($taskRecord.output)
        if ($taskName -notlike '*.candidates.jsonl' -and $taskName -notlike '*.decisions.jsonl' -and $taskName -like 'UltraData*.jsonl') {
            Add-TaskCacheFile $taskRecord.output $taskL3 'Rebuildable L3 filtering/rescue intermediate; raw and audited releases retained'
        }
    }
}
$taskReplay = Join-Path $taskDataset 'coder_replay_identity_review'
foreach ($taskFile in Get-ChildItem -LiteralPath $taskReplay -Filter 'part_*.clean.jsonl' -File) {
    Add-TaskCacheFile $taskFile.FullName $taskReplay 'Abandoned old-replay route intermediate; original source and audit evidence retained'
}
$taskPool = Join-Path $taskDataset 'cpt_unused_general_v1\pool'
foreach ($taskFile in Get-ChildItem -LiteralPath $taskPool -Filter '*.jsonl' -File) {
    if ($taskFile.Name -notlike '*.identity.jsonl') {
        Add-TaskCacheFile $taskFile.FullName $taskPool 'Rebuildable novel-source selection pool; upstream and final mixed corpus retained'
    }
}
Add-TaskCacheDirectory (Join-Path $taskWorkspace '.cache\torch_compile') 'Disposable generated torch compilation cache'
Add-TaskCacheDirectory (Join-Path $taskWorkspace '.cache\huggingface\datasets') 'Disposable Arrow/dataset loading cache; HF credentials preserved'
Add-TaskCacheDirectory (Join-Path $taskWorkspace '.pytest_cache') 'Disposable pytest cache'
foreach ($taskTop in @('model','trainer','scripts','dataset','tests','experiments')) {
    $taskTopPath = Join-Path $taskWorkspace $taskTop
    foreach ($taskDirectory in Get-ChildItem -LiteralPath $taskTopPath -Recurse -Directory -Filter '__pycache__') {
        Add-TaskCacheDirectory $taskDirectory.FullName 'Disposable Python bytecode'
    }
}

$taskManifestPath = Join-Path $taskDataset 'cpt_cache_cleanup_20261015.report.json'
$taskBeforeFree = (Get-PSDrive -Name D).Free
$taskTotalBytes = [long](($taskTargets | Measure-Object bytes -Sum).Sum)
$taskManifest = [ordered]@{
    state='prepared';created_at=(Get-Date -Format o);workspace=$taskWorkspace;
    final_parquet=$taskRelease;final_parquet_sha256=$taskAudit.sha256;
    target_count=$taskTargets.Count;planned_bytes=$taskTotalBytes;targets=@($taskTargets.ToArray());
    preserved=@('final Parquet and source JSONL','pure L3 source JSONL','all original upstream/downloaded source shards',
                'all provenance and audit reports','review candidates, manual decisions and exclusion indexes',
                'historical training corpora/checkpoints/model weights','credentials and API configuration');
    reconstruction_note='Deleted files are derived intermediates. Source and audit evidence remain; cache-dependent assembly requires regenerating removed shards.'
}
$taskManifest | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $taskManifestPath -Encoding utf8
foreach ($taskTarget in $taskTargets) {
    if ($taskTarget.kind -eq 'directory') {
        # The absolute path and link boundaries were checked above.
        Remove-Item -LiteralPath $taskTarget.path -Recurse -Force
    } else {
        Remove-Item -LiteralPath $taskTarget.path -Force
    }
}
foreach ($taskRequired in @($taskRelease, ($taskRelease + '.report.json'), ($taskRelease + '.verification.json'),
                            [IO.Path]::ChangeExtension($taskRelease, '.jsonl'))) {
    if (-not (Test-Path -LiteralPath $taskRequired -PathType Leaf)) { throw "Protected artifact disappeared: $taskRequired" }
}
$taskManifest.state = 'complete'
$taskManifest.completed_at = Get-Date -Format o
$taskManifest.removed_bytes = $taskTotalBytes
$taskManifest.free_bytes_before = $taskBeforeFree
$taskManifest.free_bytes_after = (Get-PSDrive -Name D).Free
$taskManifest | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $taskManifestPath -Encoding utf8
[pscustomobject]@{RemovedTargets=$taskTargets.Count;RemovedGB=[math]::Round($taskTotalBytes / 1e9, 2);Manifest=$taskManifestPath}
