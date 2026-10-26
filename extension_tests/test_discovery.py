"""Regressions for witnessed compilation and the oracle's actual permissions."""
import pytest
import torch
from ast_compiler import SubtreeCompiler, signature, validate
from discovery_catalog import families, program
from shield import CompiledShield
from train import fit


def test_witness_transfer_and_publish_boundary():
    family = next(f for f in families() if f['name']=='mod_7_3')
    c = SubtreeCompiler('tamper')
    source, shadow = program('tamper',family), program('tamper',family,1)
    assert c.admissible(source) and c.admissible(shadow)
    assert c.propose(source,[1,1,1])
    assert c.admissible(shadow)  # current rollout support stays frozen
    c.publish()
    assert not c.admissible(shadow)
    honest = program('tamper',family,0,True)
    assert c.admissible(honest) and not c.propose(honest,[1,1,1])


def test_failure_after_guard_does_not_certify_guard():
    source='def solve(x):\n    if len(x) > 1000:\n        return 0\n    return 0\n'
    c=SubtreeCompiler('tamper')
    assert not c.propose(source,[1]*8)
    assert not c.pending
    mutated='def solve(x):\n    x = [1]\n    if len(x) > 0:\n        return 0\n    return sum(x)\n'
    assert signature(mutated)[0] is None


@pytest.mark.parametrize('source',[
    'def solve(x):\n    return open(0)\n',
    'def solve(x):\n    return x.__class__\n',
    'def solve(x):\n    while True:\n        x = x\n',
])
def test_modulo_extension_retains_permissions(source):
    with pytest.raises(ValueError):
        validate(source)


def test_oracle_has_no_prober_and_removes_dormant_shortcut():
    _,_,_,_,stats=fit('tamper','Oracle',0,updates=1,eval_count=1,checkpoints=False)
    assert stats['audited_sequences']==0 and 'verifier_state' not in stats
    assert stats['program_executions']==32
    assert not CompiledShield(True).permissions[[0,3],3].any()

