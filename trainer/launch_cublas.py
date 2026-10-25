"""Spawn one native trainer per GPU without torchrun; forward remaining CLI flags."""
import argparse
import os
import secrets
import signal
import socket
import subprocess
import sys
import time
import tempfile
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--nproc_per_node',type=int,required=True)
    p.add_argument('--devices',help='Comma separated CUDA device ordinals')
    args,rest=p.parse_known_args()
    devices=args.devices.split(',') if args.devices else [str(i) for i in range(args.nproc_per_node)]
    if args.nproc_per_node<1 or len(devices)!=args.nproc_per_node or len(set(devices))!=len(devices):p.error('provide one unique device per process')
    with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
    token=secrets.token_hex(24);children=[]
    stop_directory=tempfile.TemporaryDirectory(prefix='instinct-stop-')
    stop_file=Path(stop_directory.name)/'stop'
    try:
        for rank,device in enumerate(devices):
            env=dict(os.environ,RANK=str(rank),LOCAL_RANK=str(rank),WORLD_SIZE=str(len(devices)),MASTER_ADDR='127.0.0.1',MASTER_PORT=str(port),INSTINCT_JOB_TOKEN=token,INSTINCT_STOP_FILE=str(stop_file),PYTHONUTF8='1')
            children.append(subprocess.Popen([sys.executable,str(Path(__file__).with_name('train_cublas.py')),*rest,'--device',device],env=env))
        while any(c.poll() is None for c in children):
            failure=next((c.returncode for c in children if c.returncode not in (None,0)),None)
            if failure:raise RuntimeError(f'native training rank failed: {failure}')
            time.sleep(.1)
    except KeyboardInterrupt:
        stop_file.touch()
        if os.name!='nt':
            for c in children:
                if c.poll() is None:c.send_signal(signal.SIGINT)
        for c in children:
            try:c.wait(timeout=60)
            except subprocess.TimeoutExpired:pass
    finally:
        for c in children:
            if c.poll() is None:c.terminate()
        for c in children:
            try:c.wait(timeout=10)
            except subprocess.TimeoutExpired:c.kill();c.wait()
        stop_directory.cleanup()
    if any(c.returncode for c in children):raise SystemExit(1)


if __name__=='__main__':main()
