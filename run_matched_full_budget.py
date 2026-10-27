"""Run the paired comparison without the earlier per-family eight-probe cap.

The budget remains exactly 44 search executions per suite/seed per method. A
family can receive up to the whole run budget, so unresolved families continue
to receive calls. The capped experiment remains available separately.
"""
import argparse
import hashlib
import json
from pathlib import Path
from run_matched_fuzzing import run

ROOT=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=ROOT/'extension_results/full_budget')
    parser.add_argument('--seeds',type=int,default=9)
    parser.add_argument('--budget',type=int,default=44)
    args=parser.parse_args()
    result=run(args.output,args.seeds,args.budget,family_cap=args.budget)
    result['protocol']['source_sha256'][Path(__file__).name]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result['protocol']['comparison'] += '; family cap equals total per-run budget'
    (args.output/'matched_fuzzing.json').write_text(json.dumps(result,indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
