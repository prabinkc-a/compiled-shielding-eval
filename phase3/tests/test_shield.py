from itertools import product
import numpy as np
import torch
from phase2.worlds import task_factory, execute, active_kernel
from phase3.shield import CompiledShield, CompilingVerifier
from torch_integration import NeuralProgramPolicy, neural_grpo_loss


def task(n=13, suite="grader"):
    return task_factory(suite, n, np.random.default_rng(71), False, False)


def test_certificate_is_versioned_and_not_ordinary_error():
    s = CompiledShield()
    t = task()
    assert not s.certify([2, 0, 0, 2, 0, 0], t, execute([2, 0, 0, 2, 0, 0], t))
    p = [0, 1, 0, 3, 0, 0]
    assert s.certify(p, t, execute(p, t))
    assert not s.compiled
    s.publish()
    assert s.compiled and s.version == 1
    assert not s.permissions[0, 3] and not s.permissions[3, 3]


def test_all_permitted_programs_exclude_exploit_for_all_guards():
    s = CompiledShield(True)
    for p in product(range(4), range(4), range(4), range(4), range(3), range(3)):
        if all(s.permissions[i, v] for i, v in enumerate(p)):
            assert p[0] != 3 and p[3] != 3 and p[4] == 0 and p[5] != 1


def test_early_exit_and_certification():
    s = CompiledShield()
    v = CompilingVerifier(4, s)
    assert not v.check([0, 1, 0, 3, 0, 0], task(8), 0)
    assert v.micro_calls == 1 and v.full_calls == 0 and s.pending


def test_valid_scale_switch_survives():
    s = CompiledShield()
    v = CompilingVerifier(4, s)
    assert v.check([0, 1, 0, 1, 0, 0], task(8), 0)
    assert not s.pending and v.full_calls == 1


def test_sampling_and_teacher_forcing_same_support_and_finite_gradients():
    torch.set_num_threads(1)
    torch.manual_seed(3)
    model = NeuralProgramPolicy()
    x = torch.tensor(np.stack([task().features()] * 8))
    shield = CompiledShield(True)
    a, old = model.generate(x, shield)
    logits, _ = model(x, a, shield)
    rescored = logits.log_softmax(-1).gather(-1, a[..., None]).squeeze(-1)
    assert torch.allclose(old, rescored, atol=1e-6)
    assert torch.isneginf(logits[:, [0, 3], 3]).all()
    loss = neural_grpo_loss(logits, a, old, torch.arange(8).float(), 8)
    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
