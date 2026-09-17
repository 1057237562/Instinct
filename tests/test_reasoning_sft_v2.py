import json
from dataset.scripts.build_reasoning_sft_v2 import rule_rows, selection_units


def test_reference_rule_curriculum():
    rows=rule_rows()
    assert len(rows)>5000
    pairs={}
    for r in rows:
        kind,a,b,c=map(int,r['group_id'].split('/')[1:]);answer=r['conversations'][1]['content']
        if kind==0:
            pairs.setdefault(r['group_id'],set()).add(answer)
            assert json.loads(answer) in ([a,c],[a,b,c])
        elif kind==1:assert json.loads(answer)==[a,a-b,a-b+c]
        elif kind==2:assert int(answer.split('####')[-1])==3*a+2*b-c
        elif kind==3:assert json.loads(answer)=={'area':a*b,'perimeter':2*(a+b)}
        else:
            import re
            q=r['conversations'][0]['content']
            cutoff=int(re.search(r'(?:大于|than )(\d+)',q).group(1))
            assert json.loads(answer)==sorted(x for x in [a,b,c] if x>cutoff and x%2==0)
    assert all(len(v)==2 for v in pairs.values())


def test_budget_units_keep_opposing_prompts_together():
    rows=[{'source':'verified_rules','group_id':'pair','tokens':7},
          {'source':'gsm8k_train','tokens':3},
          {'source':'verified_rules','group_id':'pair','tokens':8}]
    units=selection_units(rows)
    assert [len(unit) for unit in units]==[2,1]
    budget=10;selected=[]
    for unit in units:
        size=sum(r['tokens'] for r in unit)
        if size<=budget:selected.extend(unit);budget-=size
    assert all(r['source']!='verified_rules' for r in selected)
