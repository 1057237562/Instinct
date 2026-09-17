"""Launch a NEW continued-SFT stage; --dry_run validates settings without training."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot',action='store_true',help='Use the approximately 1M-token pilot subset')
    parser.add_argument('--dry_run',action='store_true')
    parser.add_argument('--learning_rate',type=float,default=1e-5)
    parser.add_argument('--batch_size',type=int,default=4)
    parser.add_argument('--accumulation_steps',type=int,default=4)
    args=parser.parse_args()
    config=ROOT/'checkpoints/full_sft_20260914_200306_768.json'
    weight=ROOT/'out/instinct-v1-0914.pth'
    dataset=ROOT/'dataset/reasoning_sft_strict'/('pilot_train.jsonl' if args.pilot else 'train.jsonl')
    for path in (config,weight,dataset):
        if not path.is_file():raise FileNotFoundError(path)
    cfg=json.loads(config.read_text(encoding='utf-8'))
    run_name='reasoning_sft_strict_'+('pilot_' if args.pilot else '')+datetime.now().strftime('%Y%m%d_%H%M%S')
    flags={'from_weight':str(weight),'from_resume':0,'config_path':str(config),
        'hidden_size':cfg['hidden_size'],'num_hidden_layers':cfg['num_hidden_layers'],'use_moe':int(cfg.get('use_moe',False)),
        'data_path':str(dataset),'save_dir':'out','save_weight':run_name,'epochs':1,
        'optimizer':'adamw','learning_rate':args.learning_rate,'batch_size':args.batch_size,
        'accumulation_steps':args.accumulation_steps,'grad_clip':1.0,
        'dtype':'bfloat16','param_dtype':'fp32','kv_cache_dtype':'fp32','fp8_training':'off',
        'max_seq_len':512,'sequence_packing':1,'sequence_packing_mode':'fixed',
        'packing_num_proc':1,'num_workers':0,'use_grad_checkpoint':1,'use_compile':0,
        'save_interval':100,'log_interval':10}
    command=[sys.executable,'-u',str(ROOT/'trainer/train_full_sft.py')]
    for k,v in flags.items():command.extend(['--'+k,str(v)])
    # Parse with the actual shared CLI without importing/executing train_full_sft.
    sys.path.append(str(ROOT))
    import datasets  # noqa: F401; before torch
    from trainer.trainer_cli import build_trainer_parser
    checker=build_trainer_parser('Validate continued SFT configuration')
    checker.add_argument('--data_path')
    parsed=checker.parse_args(command[3:])
    from trainer.trainer_utils import config_from_args
    effective=config_from_args(parsed)
    assert effective.num_hidden_layers==cfg['num_hidden_layers']
    assert effective.hidden_size==cfg['hidden_size']
    assert effective.param_dtype=='fp32'
    print(subprocess.list2cmdline(command),flush=True)
    if args.dry_run:
        print('Validated against trainer parser and model configuration. Training NOT started.')
        return
    metadata=ROOT/'checkpoints'/f'{run_name}_launch.json'
    metadata.write_text(json.dumps({'command':command,'settings':flags},indent=2),encoding='utf-8')
    subprocess.run(command,cwd=ROOT,env={**os.environ,'PYTHONUTF8':'1'},check=True)


if __name__=='__main__':main()
