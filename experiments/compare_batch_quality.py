"""Short-run non-regression check for an explicitly authorized batch change.

Equal update counts, not equal token counts: report that difference explicitly.
This does not establish convergence equivalence at an equal data/compute budget.
"""
import argparse
import json
import math
import statistics
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('reference',type=Path)
    p.add_argument('candidate',type=Path)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    a,b=(json.loads(path.read_text(encoding='utf-8')) for path in (args.reference,args.candidate))
    qa,qb=a['quality'],b['quality']
    assert a['config']==b['config']
    assert a['corpus_sha256']==b['corpus_sha256']
    assert qa['initial_parameter_sha256']==qb['initial_parameter_sha256']
    assert qa['validation_indices']==qb['validation_indices'] and qa['initial']==qb['initial']
    la,lb=qa['training_losses'],qb['training_losses']
    assert len(la)==len(lb)
    windows=[]
    for start in range(0,len(la),20):
        ra,rb=statistics.mean(la[start:start+20]),statistics.mean(lb[start:start+20])
        windows.append(dict(first_step=start+1,last_step=min(start+20,len(la)),
                            reference_mean=ra,candidate_mean=rb,relative_difference=(rb-ra)/ra))
    va,vb=qa['final']['token_weighted_loss'],qb['final']['token_weighted_loss']
    finite=all(math.isfinite(x) for x in la+lb+[va,vb])
    val_delta=(vb-va)/va
    result=dict(passed=finite and val_delta<=.01 and max(w['relative_difference'] for w in windows)<=.02,
        kernel_only_parity_metadata_not_applied=True,
        acceptance='No >2% increase in 20-update training mean or >1% increase in common validation NLL; improvements allowed',
        training_updates=len(la),same_initial_parameters=True,same_validation_examples=True,
        same_training_batches=qa['training_batch_sha256']==qb['training_batch_sha256'],
        reference_training_tokens=a['data_config']['streaming_processed_tokens'],
        candidate_training_tokens=b['data_config']['streaming_processed_tokens'],
        training_windows=windows,reference_validation_nll=va,candidate_validation_nll=vb,
        validation_relative_difference=val_delta,all_finite=finite,
        limitation='Authorized batch/packing change. Same updates but different training tokens/order; not an equal-token convergence proof.')
    args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
    if not result['passed']:
        raise SystemExit(1)


if __name__=='__main__':main()
