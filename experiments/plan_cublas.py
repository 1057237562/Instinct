"""Check WebUI config coverage without importing CUDA, Triton or Torch.

python -m experiments.plan_cublas --config_path trainer/config_pretrain.json
Optional --training_path reads WebUI panel settings (train_type/optimizer/etc.).
--require_ready returns 2 until all requested training features are implemented
and validated; a successful planning command alone does NOT mean trainable.
"""
import argparse
import json
from pathlib import Path

from model.dense_cublas.config import WebUIRequest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config_path', type=Path, required=True)
    parser.add_argument('--training_path', type=Path)
    parser.add_argument('--require_ready', action='store_true')
    args = parser.parse_args()
    try:
        model = json.loads(args.config_path.read_text(encoding='utf-8'))
        training = json.loads(args.training_path.read_text(encoding='utf-8')) if args.training_path else None
        request = WebUIRequest.from_dicts(model, training)
        report = request.report()
    except (OSError, ValueError) as exc:
        print(json.dumps({'status': 'invalid_config', 'error': str(exc)}, ensure_ascii=False, indent=2))
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 2 if args.require_ready and not report['training_ready'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
