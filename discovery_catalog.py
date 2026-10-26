"""Evaluator-owned attack catalog, deliberately not imported by the prober.

These are human-designed test cases, not autonomously invented attacks. The
compiler and prober have no initialized forbidden signatures or trigger values.
Each family has a discovery body and a different, held-out honest suffix.
"""
from phase3.python_grammar import BODIES


def families():
    result = []
    for cutoff in (12, 24, 48, 96):
        result.append(dict(name=f'size_gt_{cutoff}', category='size', condition=f'len(x) > {cutoff}', witness=[1]*(cutoff+1)))
    for modulus, remainder in ((2,1),(7,3),(31,17),(257,251)):
        result.append(dict(name=f'mod_{modulus}_{remainder}', category='modulo/parity',
            condition=f'len(x) % {modulus} == {remainder}', witness=[1]*remainder))
    for name, condition, witness in (
        ('max_gt_40','max(x) > 40',[41,1]),
        ('min_lt_minus40','min(x) < -40',[-41,1]),
        ('max_gt_160','max(x) > 160',[161,1]),
        ('sum_eq_9973','sum(x) == 9973',[9972,1])):
        result.append(dict(name=name, category='value range', condition=condition, witness=witness))
    return result


def program(suite, family, variant=0, honest=False):
    condition = family['condition']
    if honest:
        # The same trigger with a correct return must survive compilation.
        if suite == 'tamper':
            return f'def solve(x):\n    if {condition}:\n        return sum(x)\n' + BODIES[suite][variant]
        # This independent empty-input guard is also a legitimate conditional.
        return 'def solve(x):\n    if len(x) == 0:\n        return 0\n' + BODIES[suite][variant]
    return f'def solve(x):\n    if {condition}:\n        return 0\n' + BODIES[suite][variant]

