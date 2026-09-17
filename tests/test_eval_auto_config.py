import json

from streamlit.testing.v1 import AppTest

from scripts.eval_webui_utils import checkpoint_config, ROOT


def test_exact_sidecar_and_precedence(tmp_path):
    (tmp_path / 'out').mkdir()
    (tmp_path / 'checkpoints').mkdir()
    config = {'hidden_size': 768, 'num_hidden_layers': 20, 'model_architecture': 'linear'}
    sidecar = tmp_path / 'checkpoints' / 'sft_768.json'
    sidecar.write_text(json.dumps(config))
    path, actual = checkpoint_config('out/sft_768.pth', tmp_path)
    assert path == str(sidecar)
    assert actual == config
    assert checkpoint_config('out/best.pth', tmp_path) == (None, None)
    local = tmp_path / 'out' / 'sft_768.json'
    local.write_text(json.dumps({**config, 'num_hidden_layers': 12}))
    assert checkpoint_config('out/sft_768.pth', tmp_path)[1]['num_hidden_layers'] == 12


def test_ui_automates_samples_and_uses_saved_architecture():
    app = AppTest.from_file(str(ROOT / 'scripts/eval_webui.py'), default_timeout=15).run()
    app.selectbox(key='choice_pass@K').select('1,5,10').run()
    command = next(code.value for code in app.code if '--num_samples' in code.value)
    assert '--num_samples 10' in command
    assert '--temperature 0.8' in command
    assert not any(box.label in ('层数', 'hidden_size', '主干', '残差', '随机种子') for box in app.selectbox)
    selected = app.selectbox(key='out_checkpoint').value
    path, metadata = checkpoint_config(selected)
    if path:
        assert '--config_path' in command
        assert any(str(metadata['num_hidden_layers']) + ' 层' in caption.value for caption in app.caption)
