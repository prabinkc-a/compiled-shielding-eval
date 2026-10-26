import sys
from pathlib import Path
import numpy as np
import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from torch_integration import certified_policy, constrained_group_loss, unified_rewards


def test_forbidden_action_has_zero_probability_and_gradient():
    logits = torch.tensor([[0.2, -0.1, 200.]], requires_grad=True, dtype=torch.float64)
    dist = certified_policy(logits, torch.tensor([[True, True, False]]))
    assert dist.probs[0, 2].item() == 0
    loss = constrained_group_loss(dist, torch.tensor([[0, 1, 0, 1]]),
                                  torch.tensor([[1., 0., 1., 0.]]))
    loss.backward()
    assert logits.grad[0, 2].item() == 0


def test_all_illegal_fails_closed():
    with pytest.raises(ValueError, match="No certified"):
        certified_policy(torch.zeros(1, 3), torch.zeros(1, 3, dtype=torch.bool))


def test_illegal_rollout_is_not_silently_rescored():
    dist = certified_policy(torch.zeros(1, 3), torch.tensor([[True, True, False]]))
    with pytest.raises(ValueError, match="outside"):
        constrained_group_loss(dist, torch.tensor([[0, 2]]), torch.ones(1, 2))


def test_autograd_matches_numpy_benchmark_update():
    logits = torch.tensor([[0.2, -0.1, 0.4]], requires_grad=True, dtype=torch.float64)
    actions = np.array([0, 1, 2, 2])
    rewards = np.array([0.9, -0.1, 0.2, 0.2])
    dist = certified_policy(logits, torch.ones(1, 3, dtype=torch.bool))
    loss = constrained_group_loss(dist, torch.tensor(actions[None]),
                                  torch.tensor(rewards[None]), entropy_weight=0)
    loss.backward()
    a = (rewards - rewards.mean()) * 4 / 3
    expected = np.bincount(actions, weights=a, minlength=3) / 4
    assert np.allclose(-logits.grad.numpy()[0], expected, atol=1e-12)


def test_gate_multiplication_and_detachment():
    proxy = torch.ones(1, 3, requires_grad=True)
    cost = torch.full((1, 3), 0.1)
    progress = torch.ones(1, 3)
    hard = torch.tensor([[True, True, False]])
    causal = torch.ones(1, 3, dtype=torch.bool)
    iso = torch.tensor([[[True, True], [True, False], [True, True]]])
    reward = unified_rewards(proxy, cost, progress, hard, causal, iso)
    assert torch.allclose(reward, torch.tensor([[1.05, 0.05, -0.1]]))
    assert not reward.requires_grad


def test_zero_variance_group_has_no_task_gradient():
    logits = torch.zeros(1, 3, requires_grad=True)
    dist = certified_policy(logits, torch.ones(1, 3, dtype=torch.bool))
    loss = constrained_group_loss(dist, torch.tensor([[0, 1, 2]]),
                                  torch.ones(1, 3), entropy_weight=0)
    loss.backward()
    assert torch.equal(logits.grad, torch.zeros_like(logits))


def test_clipped_loss_is_finite_and_differentiable():
    logits = torch.zeros(1, 2, requires_grad=True)
    policy = certified_policy(logits, torch.ones(1, 2, dtype=torch.bool))
    actions = torch.tensor([[0, 1]])
    old = policy.logits.gather(-1, actions).detach()
    loss = constrained_group_loss(policy, actions, torch.tensor([[1., 0.]]), old_logp=old)
    loss.backward()
    assert torch.isfinite(logits.grad).all()
