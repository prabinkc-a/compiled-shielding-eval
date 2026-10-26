"""Run a real TRL training step on a local, randomly initialized tiny GPT-2.

No pretrained model or tokenizer download is needed. This validates adapter API
and gradients, not Qwen/Llama alignment performance.
"""
import json
from pathlib import Path
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
from trl import GRPOConfig
from .trl_patch import AlgorithmFGRPOTrainer
from .trl_backend import DSLBackend, make_prompt


def main():
    torch.set_num_threads(1)
    torch.manual_seed(45)
    root = Path(__file__).resolve().parent / "results" / "trl_smoke"
    root.mkdir(exist_ok=True, parents=True)
    vocab = {"[PAD]": 0, "[BOS]": 1, "[EOS]": 2, "[UNK]": 3,
             "0": 4, "1": 5, "2": 6, "3": 7, "TASK": 8, "Program": 9}
    tokenizer = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    processing = PreTrainedTokenizerFast(tokenizer_object=tokenizer, pad_token="[PAD]",
                                         bos_token="[BOS]", eos_token="[EOS]", unk_token="[UNK]")
    config = GPT2Config(vocab_size=len(vocab), n_positions=1024, n_embd=32, n_layer=1,
                        n_head=2, bos_token_id=1, eos_token_id=2, pad_token_id=0,
                        resid_pdrop=0, embd_pdrop=0, attn_pdrop=0)
    model = GPT2LMHeadModel(config)
    bias = torch.full((len(vocab),), -8.0)
    bias[4:8] = torch.tensor([2., 1., 0., -1.])
    model.lm_head.bias = torch.nn.Parameter(bias)
    backend = DSLBackend(21)
    args = GRPOConfig(output_dir=str(root), max_steps=2, num_generations=4,
                      per_device_train_batch_size=4, gradient_accumulation_steps=1,
                      max_completion_length=6, generation_kwargs={"min_new_tokens": 6},
                      loss_type="grpo", beta=0.0, learning_rate=0.001,
                      use_cpu=True, bf16=False, fp16=False, gradient_checkpointing=False,
                      dataloader_pin_memory=False, report_to="none", save_strategy="no",
                      disable_tqdm=True, logging_steps=1)
    trainer = AlgorithmFGRPOTrainer(model=model, backend=backend, args=args,
                                   processing_class=processing, train_dataset=backend.dataset(count=8),
                                   audit_floor=1.0)
    # Exercise an actual scale-regime counterexample before training, then test
    # that it is revoked and used by the auxiliary replay path during training.
    bad = trainer.algorithm_f_reward([make_prompt()], ["0 1 0 3 0 0"])
    assert bad[0] < -3
    initial = model.transformer.h[0].attn.c_attn.weight.detach().clone()
    outcome = trainer.train()
    changed = not torch.equal(initial, model.transformer.h[0].attn.c_attn.weight.detach())
    assert changed and trainer.f_metrics["replay_updates"] > 0
    payload = {"status": "passed", "steps": trainer.state.global_step,
               "model": "random local GPT2, 1 layer / 32 hidden / 2 heads",
               "parameters": sum(p.numel() for p in model.parameters()),
               "weights_changed": changed, "revoked_counterexample_reward": bad[0],
               "metrics": trainer.f_metrics, "backend_executed": backend.executed,
               "backend_blocked_before_execution": backend.blocked,
               "training_loss": outcome.training_loss}
    (root / "result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
