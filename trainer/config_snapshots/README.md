# Instinct V2 checkpoint config snapshots

These JSON files preserve the model configuration sidecars saved alongside local Instinct V2 Dense checkpoints. Filenames mirror the corresponding local checkpoint names. Some snapshots are identical because the same configuration was saved with multiple checkpoints; keep each filename for checkpoint-to-config traceability.

The shared preset and canonical model configuration remain `scripts/config_template_defaults.json` and `trainer/config_instinct_v2.json`. These snapshots record checkpoint-specific saved settings and do not replace either source. Model weights and full resume states are not included.
