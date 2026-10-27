"""One-command fresh training reproduction in a separate local output tree.

Default: rebuild historical main/ablation experiments and the new controls,
discovery study and report. --smoke exercises the pipeline cheaply; it does not
reproduce headline metrics. Existing evidence files are never overwritten.
"""
import argparse
import datetime
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--verify-only',action='store_true')
    args=parser.parse_args()
    if args.verify_only:
        subprocess.run([sys.executable,str(ROOT/'verify_release.py')],cwd=ROOT,check=True)
        return
    if args.smoke:
        subprocess.run([sys.executable,'-m','pytest','-q','tests','phase2/tests','phase3/tests','extension_tests'],cwd=ROOT,check=True)
        from trl_example import main as smoke
        smoke(args.output or ROOT/'reproduced/trl_smoke')
        print('Smoke passed; this does not reproduce the full research tables.')
        return
    dest=(args.output or ROOT/'reproduced'/datetime.datetime.now().strftime('%Y%m%d-%H%M%S')).resolve()
    if dest.exists():
        raise SystemExit('Output must be a new directory; refusing to overwrite evidence.')
    # Copy only source/config, not frozen results: every displayed result must be
    # created afresh. Historical manifests are not applied to fresh timings.
    ignored=shutil.ignore_patterns('.git','.venv','__pycache__','.pytest_cache','reproduced',
        'results','extension_results','pilot*','*.zip','SHA256SUMS.txt','MANIFEST.sha256')
    shutil.copytree(ROOT,dest,ignore=ignored)
    commands=[
        ['run_benchmark.py'],
        ['-m','phase2.run'],['-m','phase2.profile'],['-m','phase2.exact_risk'],
        ['-m','phase2.legacy_bridge'],
        ['-m','phase3.run'],['-m','phase3.profile'],['-m','phase3.exact_risk'],
        ['-m','phase3.latency'],['-m','phase3.python_trl_benchmark'],
        ['run_controls.py'],['run_discovery.py'],['run_matched_fuzzing.py'],
        ['run_matched_full_budget.py'],['audit_search.py'],['build_release_report.py'],
        ['-m','pytest','-q','tests','phase2/tests','phase3/tests','extension_tests'],
        ['trl_example.py'],
    ]
    runs=[]
    for command in commands:
        print('Running: '+sys.executable+' '+' '.join(command),flush=True)
        start=time.perf_counter()
        proc=subprocess.run([sys.executable,*command],cwd=dest,check=False)
        runs.append(dict(command=command,returncode=proc.returncode,seconds=time.perf_counter()-start))
        (dest/'reproduction_log.json').write_text(json.dumps(runs,indent=2),encoding='utf-8')
        if proc.returncode:
            raise SystemExit(f'Stage failed; preserved diagnostics in {dest}')
    subprocess.run([sys.executable,'make_manifest.py'],cwd=dest,check=True)
    subprocess.run([sys.executable,'verify_release.py'],cwd=dest,check=True)
    print(f'Full fresh reproduction complete: {dest}')


if __name__=='__main__':
    main()
