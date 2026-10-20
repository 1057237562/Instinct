# Zero-valued options use the shipped Instinct V2 recipe. Validate GPU memory before increasing them.
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$DataPath,
    [string]$Python = 'python',
    [ValidateRange(0, 2147483647)]
    [int]$MaxSeqLen = 0,
    [ValidateRange(0, 1024)]
    [int]$BatchSize = 0,
    [ValidateRange(0, 65536)]
    [int]$AccumulationSteps = 0,
    [ValidateRange(0, 10000)]
    [int]$Epochs = 0
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$configPath = Join-Path $repoRoot 'trainer/config_instinct_v2.json'
$config = Get-Content -LiteralPath $configPath -Raw | ConvertFrom-Json
$catalog = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'config_template_defaults.json') -Raw | ConvertFrom-Json
$recipe = @{}
foreach ($section in @($catalog.defaults, $catalog.presets.'Instinct V2', $catalog.templates.'Instinct V2')) {
    foreach ($property in $section.PSObject.Properties) {
        $recipe[$property.Name] = $property.Value
    }
}
if ($MaxSeqLen -eq 0) { $MaxSeqLen = $recipe.max_seq_len }
if ($BatchSize -eq 0) { $BatchSize = $recipe.batch_size }
if ($AccumulationSteps -eq 0) { $AccumulationSteps = $recipe.accumulation_steps }
if ($Epochs -eq 0) { $Epochs = $recipe.epochs_pretrain }
if ($MaxSeqLen -gt $config.max_position_embeddings) {
    throw 'MaxSeqLen exceeds the configured native context length.'
}
# Resolve a relative corpus path from the caller's directory before changing cwd.
$resolvedDataPath = (Resolve-Path -LiteralPath $DataPath).Path
$trainingArgs = @(
    'trainer/train_pretrain.py',
    '--config_path', $configPath,
    '--hidden_size', [string]$config.hidden_size,
    '--num_hidden_layers', [string]$config.num_hidden_layers,
    '--model_architecture', [string]$config.model_architecture,
    '--use_moe', '0',
    '--use_grad_checkpoint', [string]$config.use_grad_checkpoint,
    '--param_dtype', [string]$config.param_dtype,
    '--dtype', [string]$recipe.activation_dtype,
    '--optimizer', [string]$recipe.optimizer,
    '--muon_mantissa', [string][int][bool]$recipe.muon_mantissa,
    '--muon_momentum_dtype', [string]$recipe.muon_momentum_dtype,
    '--muon_workspace_mb', [string]$recipe.muon_workspace_mb,
    '--fp8_training', [string]$recipe.fp8_training,
    '--use_compile', [string][int][bool]$recipe.use_compile,
    '--device', 'cuda:0',
    '--from_weight', 'none',
    '--from_resume', '0',
    '--save_weight', 'pretrain_instinct_v2',
    '--data_path', $resolvedDataPath,
    '--max_seq_len', [string]$MaxSeqLen,
    '--batch_size', [string]$BatchSize,
    '--accumulation_steps', [string]$AccumulationSteps,
    '--epochs', [string]$Epochs,
    '--learning_rate', [string]$recipe.learning_rate_pretrain,
    '--warmup_ratio', [string]$recipe.warmup_ratio_pretrain,
    '--num_workers', [string]$recipe.num_workers
)

$previousPythonUtf8 = $env:PYTHONUTF8
$exitCode = 1
Push-Location -LiteralPath $repoRoot
try {
    $env:PYTHONUTF8 = '1'
    & $Python @trainingArgs
    $exitCode = $LASTEXITCODE
}
finally {
    $env:PYTHONUTF8 = $previousPythonUtf8
    Pop-Location
}
exit $exitCode
