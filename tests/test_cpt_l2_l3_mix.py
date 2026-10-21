"""Protect source diversity, exclusion identity, and the new token budget."""
import sys
from pathlib import Path
import uuid
import numpy as np
import scripts
scripts.__path__=[str(Path(__file__).resolve().parents[1]/'scripts'),*scripts.__path__]
from scripts.data_builder.build_cpt_l2_l3_mix import (
    L2_QUOTAS,L3_QUOTAS,GENERAL,contains,fingerprint,uid,credential_literal,
)
from scripts.data_builder.collect_cpt_l2 import quality_reason

def source(content):
    return {'content':content,'uuid':str(uuid.UUID(int=1)),'repo_name':'sample/repo',
            'relative_path':'src/algorithm.py','category':'ALGO','algo_rel_score':0.95,'quality_score':5}

def test_preserves_abstract_interfaces_and_ordinary_todos():
    r=source('class Base:\n    def run(self):\n        raise NotImplementedError\n# TODO: support more backends\n')
    assert quality_reason(r) is None

def test_separator_in_string_is_not_a_merge_conflict():
    assert quality_reason(source('banner = """\n=======\nexample\n"""\n')) is None
    assert quality_reason(source('<<<<<<< HEAD\nx = 1\n=======\nx = 2\n>>>>>>> branch\n'))=='unresolved_merge_conflict'

def test_invalid_metadata_cannot_enter_pool():
    r=source('x=1');r['quality_score']=float('nan')
    assert quality_reason(r)=='nonfinite_score'
    r=source('x=1');r['uuid']='invalid'
    assert quality_reason(r)=='invalid_source_scores_or_uuid'
    r=source('x=1');r['category']='EXCLUDE'
    assert quality_reason(r)=='upstream_quality_gate'

def test_preserves_upstream_language_specific_relevance_selection():
    r=source('def f(x):\n    return x + 1\n');r['algo_rel_score']=0.01
    assert quality_reason(r) is None

def test_exclusion_preserves_case_and_internal_whitespace():
    assert fingerprint('  x = 1\n')==fingerprint('x = 1')
    assert fingerprint('X = 1')!=fingerprint('x = 1')
    assert fingerprint('s="a b"')!=fingerprint('s="ab"')
    a=np.sort(np.array([fingerprint('x'),fingerprint('y')],dtype='V16'))
    assert contains(a,fingerprint('x')) and not contains(a,fingerprint('z'))

def test_uuid_linkage_ignores_spelling_but_not_identity():
    assert uid('ABCDEFAB-1234-1234-1234-ABCDEFABCDEF')==uid('abcdefab123412341234abcdefabcdef')
    assert uid(str(uuid.UUID(int=1)))!=uid(str(uuid.UUID(int=2)))

def test_fixed_budget_and_primary_language_share():
    assert sum(L2_QUOTAS.values())==3_600_000_000
    assert sum(L3_QUOTAS.values())==7_200_000_000
    assert sum(GENERAL.values())==1_200_000_000
    assert L2_QUOTAS['py']+L2_QUOTAS['cpp']==sum(L2_QUOTAS.values())*.75

def test_credential_screen_distinguishes_placeholders():
    assert not credential_literal('api_key = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"')
    assert not credential_literal('api_key = os.environ["OPENAI_API_KEY"]')
    assert credential_literal('api_key = "sk-'+'0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'+'"')
    assert credential_literal('-----BEGIN PRIVATE KEY-----')
