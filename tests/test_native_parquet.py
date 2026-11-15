"""Native Parquet row/target conservation, deterministic packing and resume."""
from collections import Counter
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from model.dense_cublas.data import Tokenizer
from model.dense_cublas.parquet_data import ParquetPretrain, PackedChunk, ROOT


def test_compact_block_labels_and_document_boundaries(tmp_path):
    ids=np.array([[1,3,2,1,4,2,0,0],[1,5,6,2,0,0,0,0]],dtype='<u2')
    ids.tofile(tmp_path/'tokens');np.array([0,2,3],dtype='<u8').tofile(tmp_path/'offsets')
    np.array([3,3,4],dtype='<u2').tofile(tmp_path/'lengths')
    manifest={'buckets':[dict(blocks=2,length=8,tokens_file='tokens',offsets_file='offsets',lengths_file='lengths')]}
    chunk=PackedChunk(tmp_path,manifest,0)
    try:
        (tokens,labels,segments),valid,useful=chunk.batch(0,np.array([0,1]))
        np.testing.assert_array_equal(tokens,ids)
        np.testing.assert_array_equal(segments,[[0,0,0,1,1,1,-1,-1],[0,0,0,0,-1,-1,-1,-1]])
        assert labels[0,3]==labels[0,0]==-100
        assert valid==10 and useful==7
    finally:chunk.close()


def test_rust_parquet_preserves_documents_and_exact_resume(tmp_path):
    binary=Path(os.environ.get('INSTINCT_DATASET_COMPILER',ROOT/'dataset_compiler/target/release/dataset_compiler.exe'))
    if not binary.is_file():pytest.skip('build the native dataset_compiler first')
    import pyarrow as pa
    import pyarrow.parquet as pq
    tokenizer=Tokenizer(str(ROOT/'model'))
    texts=['hello code '*i for i in range(1,17)]+['', 'def foo():\n    return 17\n']
    path=tmp_path/'source.parquet'
    counts=[len(tokenizer.encode(x))+2 for x in texts]
    pq.write_table(pa.table({'text':texts,'token_count':counts}),path,row_group_size=9)
    data=ParquetPretrain(path,tokenizer,tmp_path/'cache',lengths=[32,64],token_budget=128,
                        threads=2,compiler=binary)
    manifests=[data.build(i) for i in range(2)]
    found=[]
    for group,manifest in enumerate(manifests):
        chunk=PackedChunk(data.cache_path(group),manifest,tokenizer.pad)
        try:
            for entry in chunk.maps:
                if entry is None:continue
                tokens,offsets,lengths=entry
                for row in range(len(tokens)):
                    start=0
                    for length in lengths[int(offsets[row]):int(offsets[row+1])]:
                        found.append(tuple(tokens[row,start:start+length].tolist()));start+=int(length)
        finally:chunk.close()
    expected=[tuple([tokenizer.bos]+tokenizer.encode(text)[:62]+[tokenizer.eos]) for text in texts]
    assert Counter(found)==Counter(expected)
    assert sum(x['rows'] for x in manifests)==len(texts)
    with data:
        batches=list(data.batches(0))
        resumed=list(data.batches(0,batches[0][0]))
    assert len(resumed)==len(batches)-1
    for (a,x),(b,y) in zip(batches[1:],resumed):
        assert a==b
        for left,right in zip(x,y):np.testing.assert_array_equal(left,right)
    assert sum(cursor['useful_targets'] for cursor,_ in batches)==sum(m['useful_targets'] for m in manifests)
    assert sum(cursor['valid_tokens'] for cursor,_ in batches)==sum(map(len,expected))
    assert batches[-1][0]['group']==2 and batches[-1][0]['batch']==0


def test_shards_preserve_global_cursors_and_repacked_batches_without_original(tmp_path):
    binary = ROOT/'dataset_compiler/target/release/dataset_compiler.exe'
    import pyarrow as pa
    import pyarrow.parquet as pq
    tokenizer = Tokenizer(str(ROOT/'model'))
    texts = ['hello code '*i for i in range(1, 18)]
    path = tmp_path/'source.parquet'
    pq.write_table(pa.table({'text':texts, 'token_count':[len(tokenizer.encode(t))+2 for t in texts]}), path, row_group_size=4)
    original = ParquetPretrain(path, tokenizer, tmp_path/'original-cache', lengths=[32,64], token_budget=128, compiler=binary)
    with original:
        batches = list(original.batches(0))
    shards = tmp_path/'shards'
    subprocess.run([str(binary), 'split', '-i', str(path), '-o', str(shards), '--target-size', '1', '--threads', '2'], check=True, capture_output=True)
    path.rename(path.with_suffix('.retained'))
    # A fresh cache forces both native planning and packing from physical shards.
    split = ParquetPretrain(shards, tokenizer, tmp_path/'split-cache', lengths=[32,64], token_budget=128, compiler=binary)
    assert split.fingerprint == original.fingerprint
    assert split.plan == original.plan
    assert split.path == original.path
    with split:
        rebuilt = list(split.batches(0))
        for cursor in (dict(group=0,batch=0), batches[1][0], dict(group=1,batch=0), dict(group=5,batch=0)):
            resumed = list(split.batches(0, cursor))
            expected = [(c,b) for c,b in batches if (c['group'],c['batch']) > (cursor['group'],cursor['batch'])]
            assert len(resumed) == len(expected)
            for (a,x),(b,y) in zip(resumed,expected):
                assert a == b
                for left,right in zip(x,y):np.testing.assert_array_equal(left,right)
    for (a,x),(b,y) in zip(batches,rebuilt):
        assert a == b
        for left,right in zip(x,y):np.testing.assert_array_equal(left,right)
    report = json.loads((shards/'manifest.json').read_text())
    report['native_plans'] = {'64':original.plan}
    (shards/'manifest.json').write_text(json.dumps(report))
    # Only the chosen physical file is needed alongside the logical manifest.
    for shard in report['shards'][1:]:
        item = shards/shard['file'];item.rename(item.with_suffix('.retained'))
    selected = ParquetPretrain(shards/report['shards'][0]['file'], tokenizer,
        tmp_path/'selected-cache', lengths=[32,64], token_budget=128, compiler=binary)
    assert selected.fingerprint == original.fingerprint
    assert selected.pause_at_shard_end and selected.group_end == 1
    with selected:
        limited = list(selected.batches(0))
    expected = [(c,b) for c,b in batches if c['group'] < 1 or (c['group']==1 and c['batch']==0)]
    assert len(limited) == len(expected)
    for (a,x),(b,y) in zip(limited,expected):
        assert a == b
        for left,right in zip(x,y):np.testing.assert_array_equal(left,right)
    first = shards/report['shards'][0]['file']
    content = bytearray(first.read_bytes());content[4] ^= 1;first.write_bytes(content)
    with pytest.raises(subprocess.CalledProcessError) as error:
        ParquetPretrain(first, tokenizer, tmp_path/'split-cache', lengths=[32,64], token_budget=128, compiler=binary)
    assert 'SHA-256 mismatch' in error.value.stderr
    content[4] ^= 1;first.write_bytes(content)
    foreign = r'D:\production\original.parquet'
    report['source_identity']['path'] = foreign
    report['native_plans']['64']['source']['path'] = foreign
    (shards/'manifest.json').write_text(json.dumps(report))
    migrated = ParquetPretrain(first, tokenizer, tmp_path/'foreign-cache',
        lengths=[32,64], token_budget=128, compiler=binary)
    assert migrated.logical_path == foreign
    assert migrated.plan['source']['path'] == foreign


@pytest.mark.skipif(os.environ.get('TRITON_INTERPRET')!='1',reason='CPU interpreter integration')
def test_native_trainer_parquet_pause_resume_matches_continuous(tmp_path):
    binary=Path(os.environ.get('INSTINCT_DATASET_COMPILER',ROOT/'dataset_compiler/target/release/dataset_compiler.exe'))
    if not binary.is_file():pytest.skip('build native dataset_compiler first')
    import pyarrow as pa
    import pyarrow.parquet as pq
    from tokenizers import Tokenizer as RawTokenizer, models, pre_tokenizers
    from trainer.train_cublas import parser,train
    from tests.test_cublas_training import HostBackend
    token_dir=tmp_path/'tokenizer';token_dir.mkdir()
    raw=RawTokenizer(models.WordLevel({'<pad>':0,'<s>':1,'</s>':2,'<unk>':3,'hello':4,'world':5,'x':6},unk_token='<unk>'))
    raw.pre_tokenizer=pre_tokenizers.Whitespace()
    raw.add_special_tokens(['<pad>','<s>','</s>'])
    raw.save(str(token_dir/'tokenizer.json'))
    (token_dir/'tokenizer_config.json').write_text(json.dumps(dict(
        bos_token='<s>',eos_token='</s>',pad_token='<pad>',chat_template='{{ bos_token }}')))
    tokenizer=Tokenizer(token_dir)
    texts=['hello world x '*i for i in range(1,9)]*4
    data=tmp_path/'train.parquet'
    pq.write_table(pa.table({'text':texts,'token_count':[len(tokenizer.encode(s))+2 for s in texts]}),data,row_group_size=8)
    config=tmp_path/'config.json'
    config.write_text(json.dumps(dict(hidden_size=8,intermediate_size=12,num_hidden_layers=1,
        head_dim=4,num_attention_heads=2,num_key_value_heads=1,vocab_size=13,
        max_position_embeddings=16,use_grad_checkpoint=0,flash_attn=True,tie_word_embeddings=True)))
    base=['--config_path',str(config),'--tokenizer_path',str(token_dir),'--data_path',str(data),
        '--from_weight','none','--dtype','fp32','--fp8','0','--batch_size','2',
        '--max_seq_len','16','--sequence_buckets','8,16','--packing','--packed_attention',
        '--packing_cache',str(tmp_path/'cache'),'--accumulation_steps','2','--num_workers','1',
        '--checkpoint_dir',str(tmp_path/'checkpoints'),'--save_dir',str(tmp_path/'out')]
    args=parser().parse_args(base+['--save_weight','resumed','--max_steps','2'])
    first=train(args,HostBackend())
    assert first['steps']==2
    shards=tmp_path/'shards'
    subprocess.run([str(binary),'split','-i',str(data),'-o',str(shards),'--target-size','1'],check=True,capture_output=True)
    args.data_path=str(shards)
    args.from_resume=True;args.max_steps=4
    resumed=train(args,HostBackend())
    direct=train(parser().parse_args(base+['--save_weight','direct','--max_steps','4']),HostBackend())
    with np.load(resumed['resume'],allow_pickle=False) as a,np.load(direct['resume'],allow_pickle=False) as b:
        for key in a.files:np.testing.assert_array_equal(a[key],b[key],err_msg=key)
    # A single shard stops safely at its boundary, flushing a partial update.
    report=json.loads((shards/'manifest.json').read_text())
    one=parser().parse_args(base+['--save_weight','one-shard','--accumulation_steps','100'])
    one.data_path=str(shards/report['shards'][0]['file'])
    boundary=train(one,HostBackend())
    assert boundary['paused'] and boundary['shard_completed']==report['shards'][0]['file']
    assert boundary['next_data_path']==str(shards/report['shards'][1]['file'])
    from model.dense_cublas import checkpoint
    saved=checkpoint.metadata(boundary['resume'])
    assert saved['progress']['epoch']==0
    assert saved['progress']['parquet_cursor']['group']==1
    assert saved['progress']['parquet_cursor']['batch']==0
    assert saved['optimizers']['model']['accumulated']==0
    # Continue the next file, keeping cumulative counters and the same epoch.
    one.from_resume=True;one.data_path=boundary['next_data_path']
    after=train(one,HostBackend())
    next_saved=checkpoint.metadata(after['resume'])
    assert after['paused'] and next_saved['progress']['parquet_cursor']['group']==2
    assert next_saved['progress']['steps']>saved['progress']['steps']
    assert next_saved['progress']['processed_tokens']>saved['progress']['processed_tokens']
