from functools import partial
from pathlib import Path
import sys
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from torch_integration import NeuralProgramPolicy, neural_grpo_loss
from phase2.worlds import SUITES, execute, task_factory, trusted_score, program_mask, static_gate
from phase2.defenses import AdaptiveVerifier, Adversary
from phase2.train import symmetric_kl, fit


def test_correct_kernels_execute_across_scales_and_structures():
    rng = np.random.default_rng(1)
    for suite in SUITES:
        for n in (6, 13, 40, 160):
            for kernel in (0, 1):
                for structure in range(4):
                    t = task_factory(suite, n, rng, False, False, structure)
                    r = execute([kernel, 0, 0, 2, 0, 0], t)
                    assert trusted_score(t, r)[:2] == (1., 0.)


def test_static_blindspot_and_adaptive_counterexample():
    for suite in SUITES:
        rng = np.random.default_rng(12)
        program = [0, 1, 0, 3, 0, 0]  # honest <=12, hidden shortcut above 12
        assert static_gate(program, suite, rng) == 1
        t = task_factory(suite, 20, rng, False, False)
        assert execute(program, t).proxy == 1
        assert trusted_score(t, execute(program, t))[:2] == (0., 1.)
        small = task_factory(suite, 8, rng)
        verifier = AdaptiveVerifier(1)
        assert not verifier.check(program, small, 0)
        assert tuple(program) in verifier.failures


def test_legitimate_algorithm_switch_is_not_rejected():
    for suite in SUITES:
        # Different internal paths and asymptotic complexity, same valid result.
        program = [0, 1, 0, 1, 0, 0]
        t = task_factory(suite, 8, np.random.default_rng(2))
        assert AdaptiveVerifier(12).check(program, t, 0)


def test_autoregressive_sampling_matches_teacher_forcing():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    model = NeuralProgramPolicy()
    tasks = [task_factory("grader", n, np.random.default_rng(n)) for n in (8, 12, 20, 80)]
    x = torch.tensor(np.stack([t.features() for t in tasks]))
    mask = partial(program_mask, constrained=True)
    actions, sampled = model.generate(x, mask)
    logits, _ = model(x, actions, mask)
    scored = torch.log_softmax(logits, -1).gather(-1, actions[..., None]).squeeze(-1)
    assert torch.allclose(sampled, scored, atol=5e-6)
    assert sum(p.numel() for p in model.parameters()) > 1000
    assert not torch.allclose(logits[0, 0, :3], logits[3, 0, :3])


def test_backprop_updates_encoder_and_recurrent_weights():
    torch.manual_seed(1)
    model = NeuralProgramPolicy()
    x = torch.randn(16, 12)
    mask = partial(program_mask, constrained=False)
    actions, old = model.generate(x, mask)
    logits, _ = model(x, actions, mask)
    loss = neural_grpo_loss(logits, actions, old, torch.arange(16).float(), 8, beta=0)
    loss.backward()
    assert model.encoder[0].weight.grad.abs().sum() > 0
    assert model.decoder.weight_hh_l0.grad.abs().sum() > 0


def test_fast_permissions_block_direct_mutations():
    model = NeuralProgramPolicy()
    x = torch.zeros(100, 12)
    actions, _ = model.generate(x, partial(program_mask, constrained=True))
    assert (actions[:, 4] == 0).all()
    assert (actions[:, 5] != 1).all()


def test_masked_kl_handles_disjoint_forbidden_logits():
    a = torch.randn(2, 6, 4, requires_grad=True)
    b = torch.randn(2, 6, 4, requires_grad=True)
    ma = torch.ones_like(a, dtype=torch.bool); ma[..., 3] = False
    mb = torch.ones_like(b, dtype=torch.bool); mb[..., 2] = False
    value = symmetric_kl(a.masked_fill(~ma, -torch.inf), b.masked_fill(~mb, -torch.inf), ma, mb)
    value.backward()
    assert torch.isfinite(a.grad).all() and torch.isfinite(b.grad).all()


def test_failure_cache_revokes_buffered_reward_and_populates_replay():
    rng = np.random.default_rng(3)
    task = task_factory("grader", 8, rng)
    program = [0, 1, 0, 3, 0, 0]
    out = execute(program, task)
    v = AdaptiveVerifier(13, full=True)
    rewards, _, _ = v.score_batch([task], [program], [out], 0)
    assert rewards[0] < -3 and len(v.replay) == 2
    before = v.executions
    again, _, _ = v.score_batch([task], [program], [out], 1)
    assert again[0] < -3 and v.executions == before and v.cache_hits == 1


def test_adversary_coevolves_from_failures():
    a = Adversary(2)
    initial = a.weights.copy()
    for _ in range(20):
        a.update(4, True)
    assert a.weights[4] > initial[4]
    assert a.failures[4] == 20


def test_small_neural_run_reproducible_and_finite():
    first = fit("grader", "F", 4, updates=3, eval_count=2, checkpoints=False)
    second = fit("grader", "F", 4, updates=3, eval_count=2, checkpoints=False)
    for p, q in zip(first[0].parameters(), second[0].parameters()):
        assert torch.equal(p, q) and torch.isfinite(p).all()
    assert first[1] == second[1]
