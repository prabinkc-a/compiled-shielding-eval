"""Runnable two-update integration check for the newly generalized TRL backend.

This is a wiring/gradient test, not an expanded neural discovery benchmark.
It downloads no model. Research-scale Qwen/Llama training has not been tested.
"""
import json
from pathlib import Path
import torch
from datasets import Dataset
from transformers import GPT2Config, GPT2LMHeadModel
from trl import GRPOConfig
from phase3.python_trl_benchmark import make_tokenizer
from discovery_catalog import families, program
from python_backend import PythonBackend
from python_grammar import PythonGrammar
from trl_patch import AlgorithmGGRPOTrainer


def main(output='reproduced/trl_smoke'):
    torch.set_num_threads(1)
    torch.manual_seed(7)
    tokenizer=make_tokenizer()
    sources=[program('tamper',f) for f in families()]
    sources += ['def solve(x):\n    return sum(x)\n']
    backend=PythonBackend('tamper',seed=7,budget=2)
    grammar=PythonGrammar(tokenizer,'tamper',backend,sources)
    model=GPT2LMHeadModel(GPT2Config(vocab_size=len(tokenizer),n_positions=1024,n_embd=32,
        n_layer=2,n_head=2,bos_token_id=tokenizer.bos_token_id,eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,resid_pdrop=0.,embd_pdrop=0.,attn_pdrop=0.))
    before=model.transformer.h[0].attn.c_attn.weight.detach().clone()
    args=GRPOConfig(output_dir=str(output),max_steps=2,num_generations=8,
        per_device_train_batch_size=8,gradient_accumulation_steps=1,steps_per_generation=1,
        max_completion_length=grammar.max_tokens,loss_type='grpo',beta=0.,learning_rate=.001,
        use_cpu=True,bf16=False,fp16=False,gradient_checkpointing=False,dataloader_pin_memory=False,
        report_to='none',save_strategy='no',disable_tqdm=True)
    trainer=AlgorithmGGRPOTrainer(model=model,args=args,backend=backend,grammar=grammar,
        processing_class=tokenizer,train_dataset=Dataset.from_dict({'prompt':['<tamper><CODE>']*16}))
    trainer.train()
    assert trainer.state.global_step==2
    assert not torch.equal(before,model.transformer.h[0].attn.c_attn.weight.detach())
    assert all(torch.isfinite(p).all() for p in model.parameters())
    result=dict(status='passed',steps=2,weights_changed=True,metrics=trainer.g_metrics,backend=backend.state())
    Path(output).mkdir(parents=True,exist_ok=True)
    (Path(output)/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result


if __name__=='__main__':
    main()
