"""Read a frozen token-transition table and run a local causal Transformer.

This module cannot discover, extract, compile, or update shield rules.
"""
import json
import torch


class FrozenPythonPolicy:
    def __init__(self, directory):
        from transformers import GPT2LMHeadModel
        self.artifact=json.loads((directory/"inference_artifact.json").read_text())
        self.nodes=[{int(k):v for k,v in n.items()} for n in self.artifact["nodes"]]
        self.model=GPT2LMHeadModel.from_pretrained(directory,local_files_only=True).eval()
        self.eos=self.artifact["eos_id"]
        self.prompt=self.artifact["prompt_ids"]
        self.programs=self.artifact["programs"]
        self.by_tokens={tuple(p["tokens"]):p for p in self.programs}

    @torch.no_grad()
    def probabilities(self):
        values=[]
        for program in self.programs:
            if not program["allowed"]:
                values.append(0.);continue
            ids=torch.tensor([self.prompt+program["tokens"]])
            logits=self.model(input_ids=ids,use_cache=False).logits[0,len(self.prompt)-1:-1].float()
            permitted=torch.zeros_like(logits,dtype=torch.bool)
            node=0
            for i,token in enumerate(program["tokens"]):
                permitted[i,list(self.nodes[node])]=True
                node=self.nodes[node][token]
            lp=logits.masked_fill(~permitted,-torch.inf).log_softmax(-1)
            values.append(float(lp.gather(-1,torch.tensor(program["tokens"])[:,None]).double().sum().exp()))
        if abs(sum(values)-1)>3e-6:
            raise ValueError("Frozen Python support does not normalize")
        return values

    @torch.no_grad()
    def sample(self, count=32):
        inputs=torch.tensor([self.prompt]*count)
        positions=[0]*count; done=[False]*count
        emitted=[[] for _ in range(count)]
        cache=None
        for _ in range(max(len(p["tokens"]) for p in self.programs)):
            out=self.model(input_ids=inputs,past_key_values=cache,use_cache=True)
            cache=out.past_key_values
            logits=out.logits[:,-1].float()
            allow=torch.zeros_like(logits,dtype=torch.bool)
            for b in range(count):
                allow[b,[self.eos] if done[b] else list(self.nodes[positions[b]])]=True
            token=torch.distributions.Categorical(logits=logits.masked_fill(~allow,-torch.inf)).sample()
            for b,k in enumerate(token.tolist()):
                if not done[b]:
                    emitted[b].append(k)
                    positions[b]=self.nodes[positions[b]][k]
                    done[b]=k==self.eos
            if all(done):
                return [self.by_tokens[tuple(p)] for p in emitted]
            inputs=token[:,None]
        raise ValueError("Model did not terminate inside the published grammar")
