import ast
import numpy as np
import pytest
import torch
from phase3.python_backend import PythonBackend, validate, branch_signatures
from phase3.python_grammar import candidates, PythonGrammar


@pytest.mark.parametrize("source", [
    "import os\ndef solve(x):\n    return 0\n",
    "def solve(x):\n    return open(0)\n",
    "def solve(x):\n    return x.__class__\n",
    "def solve(x):\n    while True:\n        x = x\n",
    "def solve(x):\n    return solve(x)\n",
    "def solve(x):\n    return [y for y in x]\n",
])
def test_forbidden_python_rejected(source):
    with pytest.raises(ValueError):
        validate(source)


def test_all_grammar_functions_parse_and_legitimate_length_guards_survive():
    for suite in ("maze", "grader", "tamper"):
        for entry in candidates(suite):
            tree = validate(entry["source"])
            assert bool(branch_signatures(tree)) == entry["shortcut"]
    valid = "def solve(x):\n    if len(x) > 12:\n        return sum(x)\n    return sum(x)\n"
    # This is a candidate signature, but it must never be compiled without a
    # failing execution. No blanket ban on length guards.
    backend = PythonBackend("tamper")
    assert backend.audit(valid, 0)
    assert not backend.pending


def test_python_counterexample_early_exit_and_signature_transfer():
    b = PythonBackend("tamper")
    entries = candidates("tamper")
    bad = next(e["source"] for e in entries if e["cutoff"] == 12)
    assert b.quick(bad, 0)[1] > .9
    assert not b.audit(bad, 0)
    assert b.micro_calls == 1 and b.full_calls == 0 and len(b.pending) == 1
    b.publish()
    for e in entries:
        assert b.admissible(e["source"])[0] == (e["cutoff"] != 12)


def test_worker_executes_three_families():
    for suite in ("maze", "grader", "tamper"):
        b = PythonBackend(suite)
        code = candidates(suite)[0]["source"]
        assert b.run(code, 20, 451)[0]


def test_bpe_sampling_matches_masked_teacher_forcing():
    from phase3.python_trl_benchmark import make_tokenizer
    from transformers import GPT2Config, GPT2LMHeadModel
    torch.set_num_threads(1)
    torch.manual_seed(4)
    tok = make_tokenizer()
    b = PythonBackend("tamper")
    g = PythonGrammar(tok, "tamper", b)
    model = GPT2LMHeadModel(GPT2Config(vocab_size=len(tok), n_positions=1024,
        n_embd=16, n_layer=1, n_head=2, resid_pdrop=0, embd_pdrop=0, attn_pdrop=0))
    model.eval()
    result = g.sample(model, ["<tamper><CODE>"] * 4)
    for p, c, old in zip(result["prompt_ids"], result["completion_ids"], result["logprobs"]):
        ids = torch.tensor([p+c])
        logits = model(input_ids=ids).logits
        masked = g.masked_logits(logits, ids, len(c))
        logp = masked.log_softmax(-1).gather(-1, torch.tensor(c)[None, :, None]).squeeze()
        assert torch.allclose(logp, torch.tensor(old), atol=2e-6)
        assert torch.isfinite(logp).all()
