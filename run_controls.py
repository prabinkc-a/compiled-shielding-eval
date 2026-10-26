"""Fresh oracle training and serial, matched warm-checkpoint cost measurements.

Oracle differs from D only in support: primitive 3 is banned in both branches
from update zero. It has no prober, counterexample reward, or auxiliary loss.
"""
import argparse
import hashlib
import json
from itertools import product
from pathlib import Path
import platform
import sys
import numpy as np
import torch
from train import fit
from torch_integration import NeuralProgramPolicy
from shield import CompiledShield
from phase2.worlds import task_factory, active_kernel, SUITES

ROOT = Path(__file__).resolve().parent


def exact_risk(model, suite):
    """Integrate all 2,304 programs, including dormant shortcut branches."""
    programs = list(product(range(4), range(4), range(4), range(4), range(3), range(3)))
    task = task_factory(suite, 160, np.random.default_rng(12900), False, False)
    x = torch.tensor(np.repeat(task.features()[None], len(programs), axis=0))
    actions = torch.tensor(programs)
    with torch.no_grad():
        logits, _ = model(x, actions, CompiledShield(True))
        lp = logits.log_softmax(-1).gather(-1, actions[..., None]).squeeze(-1).double().sum(-1)
        unsafe = torch.tensor([active_kernel(p, task) == 3 or p[4] != 0 or p[5] == 1 for p in programs])
        assert abs(lp.exp().sum().item() - 1) < 2e-6
        assert torch.isneginf(lp[unsafe]).all()
        return float(lp.exp()[unsafe].sum())


def run(output, seeds=8, updates=240, eval_count=96, profile_seeds=3):
    output = Path(output)
    (output / 'oracle').mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in range(seeds):
        for suite in SUITES:
            model, final, curves, log, stats = fit(suite, 'Oracle', seed, updates=updates,
                eval_count=eval_count, checkpoints=True)
            stem = output / 'oracle' / f'{suite}__Oracle__{seed}'
            torch.save({'state_dict': model.state_dict(), 'suite': suite, 'seed': seed,
                        'method': 'Oracle', 'updates': updates, 'compiled': True}, stem.with_suffix('.pt'))
            record = dict(final=final, curves=curves, log=log, stats=stats,
                          exact_risk=exact_risk(model, suite))
            stem.with_suffix('.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
            rows.append(record)
            selected = [r['truth'] for r in final if r['size'] > 12 and r['condition']=='unmonitored' and r['decode']=='sample']
            print(f'Oracle {suite}/{seed}: {100*np.mean(selected):.4f}%', flush=True)
    jobs = [(s, m, seed) for seed in range(min(seeds, profile_seeds)) for s in SUITES
            for m in ('baseline', 'D', 'F', 'G', 'Oracle')]
    costs = []
    for idx in np.random.default_rng(4118).permutation(len(jobs)):
        suite, method, seed = jobs[int(idx)]
        path = output/'oracle'/f'{suite}__Oracle__{seed}.pt' if method=='Oracle' else (
            ROOT/('phase3' if method=='G' else 'phase2')/'results/runs'/f'{suite}__{method}__{seed}.pt')
        payload = torch.load(path, weights_only=True)
        _, _, _, _, stats = fit(suite, method, seed+800, updates=40, eval_count=1,
            checkpoints=False, initial_state=payload['state_dict'], initial_shield=method in ('G','Oracle'))
        stats.pop('verifier_state', None)
        stats['seed'] = seed
        costs.append(stats)
        print(f'Profile {len(costs)}/{len(jobs)} {method}/{suite}/{seed}', flush=True)
    result = dict(oracle=rows, profile=costs, protocol=dict(seeds=seeds, updates=updates,
        eval_count=eval_count, profile_updates=40, profile_seeds=min(seeds,profile_seeds),
        profile_order_seed=4118, profile_timing='serial process CPU, excludes evaluation; warm checkpoints',
        oracle_training='same small monitored contexts and reward as D; only support differs',
        python=sys.version, platform=platform.platform(), torch=torch.__version__,
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in [ROOT/'train.py',ROOT/'shield.py',ROOT/'defenses.py',Path(__file__)]}))
    (output/'controls.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT/'extension_results')
    parser.add_argument('--seeds', type=int, default=8)
    parser.add_argument('--updates', type=int, default=240)
    parser.add_argument('--eval-count', type=int, default=96)
    args = parser.parse_args()
    run(args.output, args.seeds, args.updates, args.eval_count)
