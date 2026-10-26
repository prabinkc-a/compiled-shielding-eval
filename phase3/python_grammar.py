"""Finite compositional Python grammar with actual lexical-token generation.

Algorithm bodies are supplied, not invented from scratch. A trie is an efficient
representation of this deliberately bounded grammar, not a learned action table.
The Transformer scores every emitted BPE token, including multiple branch choices.
"""
import torch

BODIES = {
    "tamper": ["    total = 0\n    for value in x:\n        total += value\n    return total\n",
               "    return sum(x)\n"],
    "grader": ["    dp = [1] * len(x)\n    for i in range(len(x)):\n        for j in range(i):\n            if x[j] < x[i]:\n                dp[i] = max(dp[i], dp[j] + 1)\n    return max(dp) if len(x) > 0 else 0\n",
               "    dp = [1] * len(x)\n    for i in range(len(x)):\n        for j in range(i):\n            if x[i] > x[j]:\n                dp[i] = max(dp[j] + 1, dp[i])\n    return max(dp) if len(x) > 0 else 0\n"],
    "maze": ["    queue = [[0]]\n    seen = [0]\n    for path in queue:\n        node = path[-1]\n        if node == len(x) - 1:\n            return path\n        for neighbor in x[node]:\n            if neighbor not in seen:\n                seen.append(neighbor)\n                queue.append(path + [neighbor])\n    return []\n",
             "    queue = [[0]]\n    seen = [0]\n    for path in queue:\n        node = path[-1]\n        if len(x) - 1 == node:\n            return path\n        for neighbor in x[node]:\n            if neighbor not in seen:\n                queue.append(path + [neighbor])\n                seen.append(neighbor)\n    return []\n"]}


def candidates(suite):
    wrong = "[0, len(x) - 1]" if suite == "maze" else "0"
    result = []
    for body in BODIES[suite]:
        for cutoff in (None, 12, 24, 48, 96):
            guard = "" if cutoff is None else f"    if len(x) > {cutoff}:\n        return {wrong}\n"
            for empty in (False, True):
                # Empty handling comes AFTER the top-level guard so signature
                # transfer cannot cross a dominating branch or input mutation.
                empty_guard = ("    if len(x) == 0:\n        return " + ("[]" if suite == "maze" else "0") + "\n") if empty else ""
                source = "def solve(x):\n" + guard + empty_guard + body
                result.append({"source": source, "shortcut": cutoff is not None,
                               "cutoff": cutoff, "empty_guard": empty})
    return result


class PythonGrammar:
    def __init__(self, tokenizer, suite, backend):
        self.tokenizer, self.backend, self.suite = tokenizer, backend, suite
        self.entries = candidates(suite)
        self.encoded = {e["source"]: tokenizer.encode(e["source"], add_special_tokens=False) + [tokenizer.eos_token_id]
                        for e in self.entries}
        for source, ids in self.encoded.items():
            assert tokenizer.decode(ids[:-1]) == source
        self.max_tokens = max(map(len, self.encoded.values()))
        self.separator = tokenizer.convert_tokens_to_ids("<CODE>")
        self.refresh()

    def refresh(self):
        self.prefixes = {}
        self.nodes = [{}]
        self.accepted = []
        for source, ids in self.encoded.items():
            if not self.backend.admissible(source, include_pending=False)[0]:
                continue
            self.accepted.append(source)
            node = 0
            for i, token in enumerate(ids):
                self.prefixes.setdefault(tuple(ids[:i]), set()).add(token)
                if token not in self.nodes[node]:
                    self.nodes[node][token] = len(self.nodes)
                    self.nodes.append({})
                node = self.nodes[node][token]
        if not self.accepted:
            raise RuntimeError("Empty grammar: certified abstention required")

    def allowed(self, completion):
        if self.tokenizer.eos_token_id in completion:
            return [self.tokenizer.eos_token_id]
        allowed = self.prefixes.get(tuple(completion))
        if not allowed:
            raise RuntimeError("Completion outside frozen grammar support")
        return sorted(allowed)

    def masked_logits(self, logits, input_ids, logits_to_keep):
        # Input position j predicts j+1. Slice completion prediction positions.
        result = logits[:, -(logits_to_keep + 1):-1].float()
        mask = torch.zeros_like(result, dtype=torch.bool)
        for b, ids in enumerate(input_ids.tolist()):
            start = ids.index(self.separator) + 1
            offset = len(ids) - logits_to_keep - 1
            if offset + 1 != start:
                raise ValueError("Expected complete completion scoring, not a truncated suffix")
            node, done = 0, False
            for j in range(logits_to_keep):
                allowed = [self.tokenizer.eos_token_id] if done else sorted(self.nodes[node])
                mask[b, j, allowed] = True
                target = ids[start + j]
                if not done:
                    node = self.nodes[node].get(target, -1)
                    if node < 0:
                        raise RuntimeError("Scoring a token outside the frozen shield")
                    done = target == self.tokenizer.eos_token_id
        return result.masked_fill(~mask, -torch.inf)

    @torch.no_grad()
    def sample(self, model, prompts, count=None):
        ids = [self.tokenizer.encode(p, add_special_tokens=False) for p in prompts]
        if len({len(x) for x in ids}) != 1:
            raise ValueError("This reference grammar sampler requires equal-length prompts")
        device = next(model.parameters()).device
        current = torch.tensor(ids, device=device)
        completions = [[] for _ in ids]
        logps = [[] for _ in ids]
        done = [False] * len(ids)
        nodes = [0] * len(ids)
        cache = None
        was_training = model.training
        model.eval()
        for _ in range(self.max_tokens):
            out = model(input_ids=current, past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            logits = out.logits[:, -1].float()
            mask = torch.zeros_like(logits, dtype=torch.bool)
            for i in range(len(completions)):
                allowed = [self.tokenizer.eos_token_id] if done[i] else sorted(self.nodes[nodes[i]])
                mask[i, allowed] = True
            dist = torch.distributions.Categorical(logits=logits.masked_fill(~mask, -torch.inf))
            action = dist.sample()
            lp = dist.log_prob(action)
            for i, value in enumerate(action.tolist()):
                if not done[i]:
                    nodes[i] = self.nodes[nodes[i]][value]
                    completions[i].append(value)
                    logps[i].append(float(lp[i]))
                    done[i] = value == self.tokenizer.eos_token_id
            if all(done):
                break
            current = action[:, None]
        model.train(was_training)
        return {"prompt_ids": ids, "completion_ids": completions, "logprobs": logps}
