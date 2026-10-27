"""Reconstruct both studies' seeded search trajectories without new executions.

Checks candidate scheduling, random arms, generated witness inputs and feedback
updates. Actual correctness of witnessed failures is checked separately by
``verify_release.py`` using isolated execution and the task oracle.
"""
import json
import random
from pathlib import Path
from adversarial_prober import InputProber
from discovery_catalog import families, program
from run_matched_fuzzing import UniformProber

ROOT=Path(__file__).resolve().parent


def audit(path):
    data=json.loads(Path(path).read_text(encoding='utf-8'))
    p=data['protocol']
    jobs=[(suite,seed,method) for suite in p['suites'] for seed in range(p['seeds'])
          for method in p['methods']]
    random.Random(p['job_order_seed']).shuffle(jobs)
    total=0
    for run,(suite,seed,method) in zip(data['runs'],jobs,strict=True):
        entries=run['records']
        assert all((r['suite'],r['seed'],r['method'])==(suite,seed,method) for r in entries)
        catalog=families()
        random.Random(17000+seed).shuffle(catalog)
        assert [r['family'] for r in entries]==[f['name'] for f in catalog]
        search=(InputProber if method=='adaptive' else UniformProber)(9000+seed)
        attempts={f['name']:0 for f in catalog}
        discovered=set()
        witnesses=iter(run['witnesses'])
        for i,step in enumerate(run['trace']):
            eligible=[f for f in catalog if f['name'] not in discovered and attempts[f['name']]<p['family_cap']]
            padding=not eligible
            candidates=eligible or catalog
            f=min(candidates,key=lambda e:attempts[e['name']])
            arm,x=search.sample()
            assert (step['call'],step['family'],step['arm'],step['padding'])==(i,f['name'],arm,padding)
            attempts[f['name']]+=1
            if not step['correct'] and f['name'] not in discovered:
                witness=next(witnesses)
                assert witness['x']==x and witness['source']==program(suite,f)
                discovered.add(f['name'])
            search.observe(arm,not step['correct'])
            total+=1
        assert next(witnesses,None) is None
        assert all(attempts[r['family']]==r['probes'] and (r['family'] in discovered)==r['discovered'] for r in entries)
        assert len(run['trace'])==p['budget_per_suite_seed']
    return dict(runs=len(jobs),search_executions=total)


if __name__=='__main__':
    print(json.dumps({name:audit(path) for name,path in (
        ('capped',ROOT/'extension_results/matched_fuzzing.json'),
        ('full_budget',ROOT/'extension_results/full_budget/matched_fuzzing.json'))},indent=2))
