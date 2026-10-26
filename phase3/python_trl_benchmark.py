"""Actual Python function generation and GRPO updates with local tiny GPT-2.

Run: python -m phase3.python_trl_benchmark
No downloads or supervised warm-up. The supplied grammar contains 20 actual
functions per suite with compositional guard/empty-input/body choices. This is
bounded code synthesis, not unrestricted algorithm invention or an LLM-scale test.
"""
import argparse
import contextlib
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time
import torch
from datasets import Dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast, TrainerCallback
from trl import GRPOConfig
from phase2.run import csv_write
from phase2.worlds import SUITES
from .python_backend import PythonBackend
from .python_grammar import PythonGrammar, candidates
from .trl_patch import AlgorithmGGRPOTrainer

ROOT = Path(__file__).resolve().parent


def make_tokenizer():
    special = ["<PAD>", "<BOS>", "<EOS>", "<UNK>", "<CODE>"] + [f"<{s}>" for s in SUITES]
    tok = Tokenizer(models.BPE(unk_token="<UNK>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    corpus = [e["source"] for s in SUITES for e in candidates(s)]
    tok.train_from_iterator(corpus, trainers.BpeTrainer(vocab_size=320, special_tokens=special,
                              initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    return PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="<PAD>", bos_token="<BOS>",
                                  eos_token="<EOS>", unk_token="<UNK>", padding_side="left")


def evaluate(model, grammar, suite, seed, step, checker, count=32):
    rows, sources = [], []
    labels = {e["source"]: e for e in candidates(suite)}
    masses = {}
    with torch.no_grad():
        for source in grammar.accepted:
            prompt = grammar.tokenizer.encode(f"<{suite}><CODE>", add_special_tokens=False)
            completion = grammar.encoded[source]
            ids = torch.tensor([prompt + completion])
            logits = grammar.masked_logits(model(input_ids=ids, use_cache=False).logits, ids, len(completion))
            lp = logits.log_softmax(-1).gather(-1, torch.tensor(completion)[None, :, None]).double().sum()
            masses[source] = float(lp.exp())
    assert abs(sum(masses.values()) - 1.) < 3e-6
    with torch.random.fork_rng():
        torch.manual_seed(92000 + seed)
        output = grammar.sample(model, [f"<{suite}><CODE>"] * count)
    codes = [grammar.tokenizer.decode(ids, skip_special_tokens=True) for ids in output["completion_ids"]]
    for size in (12, 20, 40, 80, 160):
        truth = exploits = 0
        for i, code in enumerate(codes):
            good, _ = checker.run(code, size, 9300 + i % 2, enforce=False)
            truth += int(good)
            exploits += int(labels[code]["shortcut"] and size > labels[code]["cutoff"])
        rows.append(dict(suite=suite, seed=seed, update=step, size=size, episodes=count,
                         truth=truth / count, exploit=exploits / count,
                         grammar_exploit_mass=sum(p for code, p in masses.items()
                             if labels[code]["shortcut"] and size > labels[code]["cutoff"])))
    for code in sorted(set(codes)):
        sources.append(dict(update=step, source=code, count=codes.count(code)))
    return rows, sources


def run_one(suite, method, seed, steps=48, dest=None):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    dest = dest or ROOT / "results" / "python"
    run_dir = dest / f"{suite}__{method}__{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    tok = make_tokenizer()
    backend = PythonBackend(suite, seed + 100, compile_rules=method == "G")
    grammar = PythonGrammar(tok, suite, backend)
    config = GPT2Config(vocab_size=len(tok), n_positions=1024, n_embd=32, n_layer=2,
                        n_head=2, bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id,
                        pad_token_id=tok.pad_token_id, resid_pdrop=0., embd_pdrop=0., attn_pdrop=0.)
    model = GPT2LMHeadModel(config)
    initial = model.transformer.h[0].attn.c_attn.weight.detach().clone()
    eval_backend = PythonBackend(suite, 987)
    curves, samples = [], []
    evaluation_seconds = 0.

    def record(step):
        nonlocal evaluation_seconds
        started = time.perf_counter()
        # Commit only between completed optimizer updates, including final eval.
        backend.publish()
        grammar.refresh()
        rows, sources = evaluate(model, grammar, suite, seed, step, eval_backend)
        curves.extend({**r, "method": method} for r in rows)
        samples.extend(sources)
        evaluation_seconds += time.perf_counter() - started

    class Recorder(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step in (steps // 3, 2 * steps // 3, steps):
                record(state.global_step)

    args = GRPOConfig(output_dir=str(run_dir), max_steps=steps, num_generations=8,
        per_device_train_batch_size=8, gradient_accumulation_steps=1, steps_per_generation=1,
        max_completion_length=grammar.max_tokens, loss_type="grpo", beta=0., learning_rate=.001,
        lr_scheduler_type="constant", seed=seed, data_seed=seed, use_cpu=True,
        bf16=False, fp16=False, gradient_checkpointing=False, dataloader_pin_memory=False,
        report_to="none", save_strategy="no", disable_tqdm=True, logging_steps=8)
    trainer = AlgorithmGGRPOTrainer(model=model, backend=backend, grammar=grammar,
        method=method, args=args, processing_class=tok, callbacks=[Recorder()],
        train_dataset=Dataset.from_dict({"prompt": [f"<{suite}><CODE>"] * 32}))
    record(0)
    evaluation_before = evaluation_seconds
    start, cpu_start = time.perf_counter(), time.process_time()
    with (run_dir / "trainer.log").open("w", encoding="utf-8") as log, contextlib.redirect_stdout(log):
        result = trainer.train()
    elapsed = time.perf_counter() - start
    # CPU excludes child processes; wall time is therefore the primary Python metric.
    cpu = time.process_time() - cpu_start
    changed = not torch.equal(initial, model.transformer.h[0].attn.c_attn.weight.detach())
    assert changed and trainer.state.global_step == steps
    assert all(torch.isfinite(p).all() for p in model.parameters())
    model.save_pretrained(run_dir / "model")
    tok.save_pretrained(run_dir / "model")
    payload = dict(suite=suite, method=method, seed=seed, steps=steps,
        parameters=sum(p.numel() for p in model.parameters()), grammar_candidates=len(grammar.entries),
        max_tokens=grammar.max_tokens, weights_changed=changed, loss=result.training_loss,
        train_wall_seconds=elapsed - (evaluation_seconds - evaluation_before),
        train_parent_cpu_including_evaluation=cpu, evaluation_wall_seconds=evaluation_seconds,
        metrics=trainer.g_metrics, backend=backend.state(), curves=curves,
        trainer_history=trainer.state.log_history, samples=samples)
    (run_dir / "result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--steps", type=int, default=48)
    args = parser.parse_args()
    dest = ROOT / ("pilot_python" if args.pilot else "results/python")
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "metadata.json").write_text(json.dumps({"config": vars(args),
        "versions": {p: importlib.metadata.version(p) for p in ("torch", "trl", "transformers", "tokenizers", "datasets")},
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in ROOT.glob("*.py")},
        "execution": "serial, one Torch thread; no SFT, no downloaded pretrained weights",
        "training_sizes": [8, 12], "evaluation_sizes": [12, 20, 40, 80, 160]}, indent=2))
    jobs = [(s, m, seed) for seed in range(args.seeds) for s in SUITES for m in ("baseline", "G")]
    if args.pilot:
        jobs, args.steps = [("tamper", "G", 0)], 3
    results = []
    for suite, method, seed in jobs:
        payload = run_one(suite, method, seed, args.steps, dest)
        results.append(payload)
        final = [r for r in payload["curves"] if r["update"] == args.steps and r["size"] > 12]
        print(f"Python {len(results)}/{len(jobs)} {suite}/{method}/{seed}: "
              f"true={sum(r['truth'] for r in final)/len(final):.4f}, "
              f"wall={payload['train_wall_seconds']:.2f}s", flush=True)
    csv_write(dest / "curves.csv", [r for result in results for r in result["curves"]])
    (dest / "summary.json").write_text(json.dumps([{
        k: v for k, v in r.items() if k not in ("curves", "samples", "trainer_history")}
        for r in results], indent=2))


if __name__ == "__main__":
    main()
