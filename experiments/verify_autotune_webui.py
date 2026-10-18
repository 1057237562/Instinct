"""Exercise the actual WebUI plan launcher without GPU training or saves."""
import json
from pathlib import Path
import time

from streamlit.testing.v1 import AppTest
from scripts.autotune_panel import ROOT, run_status, write_json


state_path = ROOT / 'trainer/webui_state.json'
backup = state_path.read_bytes() if state_path.exists() else None
output = ROOT / 'experiments/results/config_webui_autotune'
try:
    write_json(state_path, dict(preset='Instinct V1 MoE', train_type='pretrain',
        sequence_packing=True, sequence_packing_mode='bucket', dataset_streaming='on',
        cache_build_mode='inline', data_path_pretrain=str(ROOT / 'dataset/pretrain_coder_12b.parquet'),
        autotune_output=str(output), autotune_max_trials=4,
        autotune_peak_tflops=0., autotune_target_tokens_per_second=0.))
    app = AppTest.from_file(str(ROOT / 'scripts/config_webui.py'), default_timeout=60).run()
    assert not app.exception
    button = next(b for b in app.button if b.key == 'btn_autotune_plan')
    assert not button.disabled
    button.click().run()
    assert not app.exception
    directory = app.session_state['autotune_run_dir']
    until = time.monotonic() + 30
    status = run_status(directory)
    while status.get('status') == 'running' and time.monotonic() < until:
        time.sleep(.5)
        status = run_status(directory)
    assert status.get('status') == 'success', status
    plan = json.loads((Path(directory) / 'controller.log').read_text(encoding='utf-8'))
    assert plan['saves_formal_weights'] is False
    assert plan['maximum_updates'] == 4 * (20 + 60)
    app.run()
    assert not app.exception
    write_json(output / 'verification.json', dict(status='verified',
        run_dir=directory, controller_status=status, plan=plan,
        webui_rerender_passed=True, gpu_training_executed=False, formal_weights_saved=False))
    print(json.dumps(dict(status='verified', run_dir=directory, maximum_updates=plan['maximum_updates']), ensure_ascii=False))
finally:
    if backup is not None:
        state_path.write_bytes(backup)
    else:
        state_path.unlink(missing_ok=True)
