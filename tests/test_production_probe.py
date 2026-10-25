"""The production throughput probe must keep the real recipe and data path."""
import json
from pathlib import Path

from experiments.profile_production_pretrain import panel_command


def test_probe_builds_real_panel_command_without_mutating_panel(tmp_path):
    state = dict(hidden_size=1536, num_hidden_layers=32, intermediate_size=5504,
                 num_attention_heads=16, num_key_value_heads=4, vocab_size=6400,
                 max_position_embeddings=4096, use_moe=False, model_architecture='standard',
                 use_grad_checkpoint=4, param_dtype='bf16', activation_dtype='bfloat16',
                 fp8_training='tensorwise', optimizer='muon', muon_workspace_mb=512,
                 sequence_packing=True, sequence_packing_mode='bucket', seq_bucket=2,
                 bucket_token_budget=12288, accumulation_steps=8, packing_num_proc=4,
                 dataset_streaming='on', cache_build_mode='inline',
                 streaming_prefetch_chunks=True, data_path_pretrain='production.parquet',
                 compile_mode='max-autotune-no-cudagraphs', learning_rate_pretrain=0.0005)
    before = json.dumps(state, sort_keys=True)
    path = tmp_path / 'model.json'
    config, command = panel_command(state, 'isolated_probe', path)
    arguments = dict(zip(command[3::2], command[4::2]))
    assert config['hidden_size'] == 1536
    assert config['head_dim'] == 96
    assert config['use_grad_checkpoint'] == 4
    for key, value in {
        '--config_path': str(path), '--data_path': 'production.parquet',
        '--from_weight': 'none', '--save_weight': 'isolated_probe',
        '--use_grad_checkpoint': '4', '--bucket_token_budget': '12288',
        '--accumulation_steps': '8', '--fp8_training': 'tensorwise',
        '--streaming_prefetch_chunks': '1', '--packing_num_proc': '4',
        '--num_workers': '0', '--learning_rate': '0.0005',
    }.items():
        assert arguments[key] == value
    assert json.dumps(state, sort_keys=True) == before
    assert not path.exists()  # Command generation alone does not write panel/config files.
