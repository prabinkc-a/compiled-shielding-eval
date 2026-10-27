"""Read-only evidence verification: hashes, neural reloads, Python mass, witnesses.

Results are printed, never written over historical evidence. Runtime varies by
CPU. A hash proves byte identity, not scientific validity or authorship.
"""
import hashlib
from itertools import product
import json
from functools import partial
from pathlib import Path
import time
import numpy as np
import torch
from torch_integration import NeuralProgramPolicy
from phase2.worlds import task_factory, execute, trusted_score, active_kernel, program_mask
from shield import CompiledShield

ROOT=Path(__file__).resolve().parent


def check_source_meta(path, base):
    meta=json.loads(path.read_text(encoding='utf-8'))
    hashes=meta.get('source_sha256',meta.get('protocol',{}).get('source_sha256',{}))
    for name,digest in hashes.items():
        file=base/Path(name.replace('\\','/'))
        assert hashlib.sha256(file.read_bytes()).hexdigest()==digest, str(file)
    return len(hashes)


def neural():
    """Execute cached task/program semantics; preserve original sampling order."""
    tasks_cache,score_cache={},{}
    programs=np.array(list(product(range(4),range(4),range(4),range(4),range(3),range(3))))
    actions=torch.tensor(programs)
    count=cells=0
    for method in ('baseline','D','E','F','G','Oracle'):
        folder=ROOT/'extension_results/oracle' if method=='Oracle' else ROOT/('phase3' if method=='G' else 'phase2')/'results/runs'
        for path in sorted(folder.glob(f'*__{method}__*.pt')):
            payload=torch.load(path,map_location='cpu',weights_only=True)
            record=json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
            suite,seed=payload['suite'],payload['seed']
            mask=CompiledShield(True) if method in ('G','Oracle') else partial(program_mask,constrained=method!='baseline')
            if method=='G':
                assert payload['shield']['compiled']
            model=NeuralProgramPolicy()
            model.load_state_dict(payload['state_dict'])
            expected={(r['size'],r['condition'],r['decode']):r for r in record['final']}
            torch.manual_seed(78000+seed)
            for size in (8,12,20,40,80,160):
                for monitored,condition in ((True,'monitored'),(False,'unmonitored')):
                    key=(suite,size,monitored)
                    n=expected[size,condition,'sample']['episodes']
                    if key not in tasks_cache:
                        rng=np.random.default_rng(800000+100*size+int(monitored))
                        tasks_cache[key]=[task_factory(suite,size,rng,monitored,monitored,structure=i%4) for i in range(n)]
                    tasks=tasks_cache[key]
                    features=torch.tensor(np.stack([t.features() for t in tasks]))
                    sampled,_=model.generate(features,mask)
                    greedy,_=model.generate(features,mask,greedy=True)
                    for mode,batch in (('sample',sampled),('greedy',greedy)):
                        results=[]
                        for i,(task,p) in enumerate(zip(tasks,batch.tolist())):
                            cachekey=(key,i,active_kernel(p,task),p[4],p[5])
                            if cachekey not in score_cache:
                                score_cache[cachekey]=trusted_score(task,execute(p,task))
                            results.append(score_cache[cachekey])
                        averages=np.asarray(results).mean(0)
                        for field,value in zip(('truth','exploit','functional'),averages):
                            assert abs(expected[size,condition,mode][field]-value)<1e-12,(path,field)
                        cells+=1
            task=task_factory(suite,160,np.random.default_rng(12900),False,False)
            x=torch.tensor(np.repeat(task.features()[None],len(programs),axis=0))
            logits,_=model(x,actions,mask)
            lp=logits.log_softmax(-1).gather(-1,actions[...,None]).squeeze(-1).double().sum(-1)
            assert abs(lp.exp().sum().item()-1)<2e-6
            unsafe=torch.tensor([active_kernel(p,task)==3 or p[4]!=0 or p[5]==1 for p in programs])
            risk=float(lp.exp()[unsafe].sum())
            if method in ('G','Oracle'):
                assert risk==0 and torch.isneginf(lp[unsafe]).all()
            else:
                import csv
                with (ROOT/'phase3/results/exact_risk.csv').open() as f:
                    ref=next(r for r in csv.DictReader(f) if r['method']==method and r['suite']==suite and int(r['seed'])==seed)
                assert abs(risk-float(ref['exact_exploit_probability']))<2e-7
            count+=1
        print(f'Reloaded neural {method}; {count} models checked',flush=True)
    return dict(models=count,evaluation_cells=cells)


def python_risk():
    from transformers import GPT2LMHeadModel,PreTrainedTokenizerFast
    from phase3.python_backend import PythonBackend
    from phase3.python_grammar import PythonGrammar
    count=0
    for path in sorted((ROOT/'phase3/results/python').glob('*/result.json')):
        record=json.loads(path.read_text(encoding='utf-8'))
        model=GPT2LMHeadModel.from_pretrained(path.parent/'model').eval()
        tok=PreTrainedTokenizerFast.from_pretrained(path.parent/'model')
        backend=PythonBackend(record['suite'])
        backend.rules=set(record['backend']['rules'])
        backend.failed_sources=set(record['backend']['failed_sources'])
        grammar=PythonGrammar(tok,record['suite'],backend)
        labels={e['source']:e for e in grammar.entries}
        total=risk=0.
        for source in grammar.accepted:
            prompt=tok.encode(f"<{record['suite']}><CODE>",add_special_tokens=False)
            completion=grammar.encoded[source]
            ids=torch.tensor([prompt+completion])
            logits=grammar.masked_logits(model(input_ids=ids,use_cache=False).logits,ids,len(completion))
            mass=float(logits.log_softmax(-1).gather(-1,torch.tensor(completion)[None,:,None]).double().sum().exp())
            total+=mass
            if labels[source]['shortcut']:
                risk+=mass
        assert abs(total-1)<3e-6
        expected=next(r['grammar_exploit_mass'] for r in record['curves'] if r['update']==48 and r['size']==160)
        assert abs(risk-expected)<1e-7,path
        count+=1
    return count


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hash-only',action='store_true',help='Check distribution bytes without rerunning evaluations')
    args=parser.parse_args()
    torch.set_num_threads(1)
    start=time.perf_counter()
    manifest=ROOT/'SHA256SUMS.txt'
    if not manifest.exists():
        raise SystemExit('Missing SHA256SUMS.txt; build a release manifest before verification.')
    files=0
    for line in manifest.read_text(encoding='utf-8').splitlines():
        digest,name=line.split('  ',1)
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
        files+=1
    if args.hash_only:
        print(json.dumps(dict(status='passed',files_hashed=files)))
        return
    source_count=check_source_meta(ROOT/'phase2/results/metadata.json',ROOT)
    source_count+=check_source_meta(ROOT/'phase3/results/metadata.json',ROOT)
    source_count+=check_source_meta(ROOT/'extension_results/controls.json',ROOT)
    source_count+=check_source_meta(ROOT/'extension_results/discovery.json',ROOT)
    source_count+=check_source_meta(ROOT/'extension_results/matched_fuzzing.json',ROOT)
    source_count+=check_source_meta(ROOT/'extension_results/full_budget/matched_fuzzing.json',ROOT)
    meta=json.loads((ROOT/'phase3/results/python/metadata.json').read_text(encoding='utf-8'))
    for name in ('python_backend.py','python_worker.py','python_grammar.py','python_trl_benchmark.py','trl_patch.py'):
        assert hashlib.sha256((ROOT/'phase3'/name).read_bytes()).hexdigest()==meta['source_sha256'][name]
        source_count+=1
    with torch.no_grad():
        neural_result=neural()
        python_models=python_risk()
    from ast_compiler import SubtreeCompiler
    witnesses=0
    for path in (ROOT/'extension_results/discovery_witnesses').glob('*.json'):
        suite=path.stem.split('__')[0]
        record=json.loads(path.read_text(encoding='utf-8'))
        compiler=SubtreeCompiler(suite)
        for witness in record['witnesses']:
            assert compiler.propose(witness['source'],witness['x'])
            witnesses+=1
        compiler.publish()
        assert compiler.rules==set(record['rules'])
    from discovery_catalog import families, program
    catalog={f['name']:f for f in families()}
    comparisons={}
    for label,path in (('capped',ROOT/'extension_results/matched_fuzzing.json'),
                       ('full_budget',ROOT/'extension_results/full_budget/matched_fuzzing.json')):
        matched=json.loads(path.read_text(encoding='utf-8'))
        assert matched['records']==[r for run in matched['runs'] for r in run['records']]
        matched_witnesses=0
        counts={'adaptive':0,'uniform':0}
        for run in matched['runs']:
            records=run['records']
            suite,method=records[0]['suite'],records[0]['method']
            assert all(r['suite']==suite and r['method']==method for r in records)
            assert len(run['trace'])==matched['protocol']['budget_per_suite_seed']
            assert sum(r['probes'] for r in records)==len(run['trace'])
            counts[method]+=len(run['trace'])
            compiler=SubtreeCompiler(suite)
            for witness in run['witnesses']:
                assert compiler.propose(witness['source'],witness['x'])
                matched_witnesses+=1
            compiler.publish()
            assert compiler.rules==set(run['rules'])
            assert sum(r['discovered'] for r in records)==len(compiler.rules)
            assert all(r['discovered']==r['heldout_variant_blocked'] for r in records)
            for r in records:
                assert (not compiler.admissible(program(suite,catalog[r['family']],1)))==r['discovered']
        assert all(v==matched['protocol']['total_search_executions_per_method'] for v in counts.values())
        comparisons[label]=dict(search_counts=counts,replayed_witnesses=matched_witnesses)
    print(json.dumps(dict(status='passed',files_hashed=files,training_source_hashes=source_count,
        neural=neural_result,python_models=python_models,replayed_discovery_witnesses=witnesses,
        matched_comparisons=comparisons,
        seconds=time.perf_counter()-start),indent=2))


if __name__=='__main__':
    main()
