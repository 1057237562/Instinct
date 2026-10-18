"""Compare paired real-corpus runs against predeclared training/validation gates."""
import argparse,json,math,statistics
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('reference',type=Path); p.add_argument('candidate',type=Path)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    a=json.loads(args.reference.read_text()); b=json.loads(args.candidate.read_text())
    qa,qb=a['quality'],b['quality']
    assert a['config']==b['config'] and a['args']==b['args']
    assert a['corpus_sha256']==b['corpus_sha256']
    assert qa['initial_parameter_sha256']==qb['initial_parameter_sha256']
    assert qa['training_batch_sha256']==qb['training_batch_sha256']
    assert qa['validation_indices']==qb['validation_indices']
    assert qa['initial']==qb['initial']
    assert qa['predeclared_acceptance']==qb['predeclared_acceptance']
    la,lb=qa['training_losses'],qb['training_losses']
    assert len(la)==len(lb)
    finite=all(math.isfinite(x) for x in la+lb)
    windows=[]
    for start in range(0,len(la),20):
        aa,bb=statistics.mean(la[start:start+20]),statistics.mean(lb[start:start+20])
        windows.append(dict(first_step=start+1,last_step=min(start+20,len(la)),
                            reference_loss=aa,candidate_loss=bb,relative_difference=abs(bb-aa)/abs(aa)))
    va,vb=qa['final']['token_weighted_loss'],qb['final']['token_weighted_loss']
    delta=(vb-va)/va
    limits=qa['predeclared_acceptance']
    passed=(finite and max(row['relative_difference'] for row in windows)<=limits['maximum_20_step_mean_loss_relative']
            and abs(delta)<=limits['final_validation_nll_relative'])
    result=dict(passed=passed,same_initial_parameters=True,same_training_batches=True,
        common_evaluator='eager BF16, eval/no_grad; historical causal SDPA for both models',
        training_steps=len(la),training_windows=windows,
        validation_examples=len(qa['final']['examples']),
        validation_tokens=sum(row['tokens'] for row in qa['final']['examples']),
        initial_validation_nll=qa['initial']['token_weighted_loss'],
        reference_final_validation_nll=va,candidate_final_validation_nll=vb,
        validation_relative_difference=delta,limits=limits,all_finite=finite,
        reference_ms=a['wall_ms_per_step'],candidate_ms=b['wall_ms_per_step'],
        speedup=a['wall_ms_per_step']/b['wall_ms_per_step'],
        limitation='A bounded 100-step training/validation comparison, not a full convergence or final-task-quality proof.')
    args.output.write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))
    assert passed


if __name__=='__main__': main()
