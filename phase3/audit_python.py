"""Reload every saved Transformer/tokenizer/shield and reproduce grammar risk."""
import hashlib
import json
from pathlib import Path
import torch
import numpy as np
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast
from phase2.worlds import task_factory
from .python_backend import PythonBackend
from .python_grammar import PythonGrammar

ROOT = Path(__file__).resolve().parent


def main():
    torch.set_num_threads(1)
    meta = json.loads((ROOT / "results/python/metadata.json").read_text())
    # Only implementation dependencies, not reports added during experiments.
    for name in ("python_backend.py", "python_worker.py", "python_grammar.py", "python_trl_benchmark.py", "trl_patch.py"):
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == meta["source_sha256"][name], name
    rows = []
    tokenizer_hashes, initial_conditions = set(), {}
    for record in sorted((ROOT / "results/python").glob("*/result.json")):
        result = json.loads(record.read_text())
        tokenizer_hashes.add(hashlib.sha256((record.parent/"model/tokenizer.json").read_bytes()).hexdigest())
        key = (result["suite"],result["seed"])
        initial = [{k:v for k,v in row.items() if k != "method"} for row in result["curves"] if row["update"]==0]
        if key in initial_conditions:
            assert initial_conditions[key] == initial
        initial_conditions[key] = initial
        model = GPT2LMHeadModel.from_pretrained(record.parent/"model").eval()
        tok = PreTrainedTokenizerFast.from_pretrained(record.parent/"model")
        backend = PythonBackend(result["suite"])
        backend.rules = set(result["backend"]["rules"])
        backend.failed_sources = set(result["backend"]["failed_sources"])
        grammar = PythonGrammar(tok, result["suite"], backend)
        labels = {e["source"]:e for e in grammar.entries}
        # Independently reconstruct witnesses: a timeout alone is insufficient
        # evidence of a semantic failure. These supplied early returns have
        # directly checkable wrong outputs on the recorded challenge input.
        witnessed = set()
        for w in result["backend"]["witnesses"]:
            entry = labels[w["source"]]
            assert entry["shortcut"] and w["size"] > entry["cutoff"]
            seed = 12000 + w["step"]
            task = task_factory(result["suite"],w["size"],np.random.default_rng(seed),False,False,seed%4)
            if result["suite"] == "maze":
                assert w["size"]-1 not in task.payload[0][0]
            else:
                assert task.target != 0
            witnessed.update(w["signatures"])
        assert backend.rules <= witnessed
        masses = {}
        with torch.no_grad():
            for source in grammar.accepted:
                prompt = tok.encode(f"<{result['suite']}><CODE>", add_special_tokens=False)
                completion = grammar.encoded[source]
                ids = torch.tensor([prompt+completion])
                logits = grammar.masked_logits(model(input_ids=ids, use_cache=False).logits, ids, len(completion))
                lp = logits.log_softmax(-1).gather(-1, torch.tensor(completion)[None,:,None]).double().sum()
                masses[source] = float(lp.exp())
        assert abs(sum(masses.values())-1) < 3e-6
        risk = sum(p for source,p in masses.items() if labels[source]["shortcut"])
        expected = next(r["grammar_exploit_mass"] for r in result["curves"] if r["update"]==48 and r["size"]==160)
        assert abs(risk-expected) < 1e-7, record
        rows.append({"run":record.parent.name,"reloaded_risk":risk,"saved_risk":expected,
                     "independently_validated_witnesses":len(result["backend"]["witnesses"])})
    assert len(tokenizer_hashes)==1
    payload = {"status":"passed","models":len(rows),"source_hashes_verified":5,
               "identical_tokenizers":True,"matched_initial_evaluations":len(initial_conditions),"rows":rows}
    (ROOT / "results/python_checkpoint_audit.json").write_text(json.dumps(payload,indent=2))
    print(f"Reloaded {len(rows)} saved Python models and reproduced finite-grammar risk")


if __name__ == "__main__":
    main()
