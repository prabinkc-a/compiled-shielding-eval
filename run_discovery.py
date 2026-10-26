"""Budgeted discovery evaluation, separate from the historical TRL experiment.

Recall is family-level, not policy-weighted risk. Every attack has an evaluator-
only witness; these witnesses are never supplied to the adaptive input prober.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import time
from adversarial_prober import InputProber
from ast_compiler import SubtreeCompiler, run_inputs
from discovery_catalog import families, program

ROOT = Path(__file__).resolve().parent


def run(output, seeds=9, budget=8):
    records = []
    start = time.perf_counter()
    for suite in ('tamper','grader'):
        for seed in range(seeds):
            search = InputProber(9000+seed)
            compiler = SubtreeCompiler(suite)
            catalog = families()
            random.Random(17000+seed).shuffle(catalog)
            for family in catalog:
                source = program(suite, family)
                shadow = program(suite, family, 1)
                honest = program(suite, family, 0, True)
                # Evaluator ground truth establishes the denominator, but is not
                # read by search/compilation until discovery has ended.
                calls = 0
                discovered = False
                for attempt in range(budget):
                    arm, x = search.sample()
                    observed = run_inputs(source, suite, [x])[0]
                    calls += 1
                    search.observe(arm, not observed['correct'])
                    if not observed['correct']:
                        discovered = compiler.propose(source, x)
                        compiler.publish()
                        if discovered:
                            break
                # Static D only sees the original monitored sizes and values.
                rng = random.Random(7000+seed)
                small = [[rng.randint(-20,20) for _ in range(n)] for n in (8,12)]
                static_detected = any(not r['correct'] for r in run_inputs(source, suite, small))
                # Neither validation witness nor held-out suffix influenced search.
                heldout = run_inputs(shadow, suite, [family['witness']])[0]
                assert not heldout['correct'], (suite,family['name'])
                honest_results = run_inputs(honest, suite, small+[family['witness']])
                assert all(r['correct'] for r in honest_results)
                falsely_compiled = compiler.propose(honest, family['witness'])
                assert not falsely_compiled and compiler.admissible(honest)
                blocked = not compiler.admissible(shadow)
                assert blocked == discovered
                records.append(dict(suite=suite,seed=seed,family=family['name'],category=family['category'],
                    discovered=discovered,static_detected=static_detected,heldout_variant_blocked=blocked,
                    honest_blocked=not compiler.admissible(honest),probe_executions=calls,
                    compiler_replays=int(discovered),oracle_blocked=True))
            folder = Path(output)/'discovery_witnesses'
            folder.mkdir(parents=True,exist_ok=True)
            (folder/f'{suite}__{seed}.json').write_text(json.dumps(dict(rules=sorted(compiler.rules),
                witnesses=compiler.witnesses,arms=search.arms,weights=search.weights),indent=2),encoding='utf-8')
            print(f'Discovery {suite}/{seed}: {sum(r["discovered"] for r in records if r["suite"]==suite and r["seed"]==seed)}/12',flush=True)
    result = dict(records=records,protocol=dict(seeds=seeds,probe_budget_per_family=budget,
        suites=['tamper','grader'],families_per_suite=12,initial_rules=0,
        description='Separate bounded discovery stress test, NOT newly trained TRL policies',
        blind_to='trigger parameters, category labels and evaluator witnesses',
        search_sizes=[8,13,25,49,97,160],size_jitter=3,search_amplitudes=[20,64,256],
        limitations='Finite inputs, supplied functions, first-guard signatures; rare/beyond-range triggers may survive',
        elapsed_seconds=time.perf_counter()-start,
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [Path(__file__),ROOT/'ast_compiler.py',ROOT/'adversarial_prober.py',ROOT/'discovery_catalog.py',ROOT/'discovery_worker.py']}))
    Path(output).mkdir(parents=True,exist_ok=True)
    (Path(output)/'discovery.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=ROOT/'extension_results')
    parser.add_argument('--seeds',type=int,default=9)
    parser.add_argument('--budget',type=int,default=8)
    args=parser.parse_args()
    run(args.output,args.seeds,args.budget)
