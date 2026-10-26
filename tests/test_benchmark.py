from itertools import product
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from environments import Maze, Grader, Tamper, PLANS, FIELDS, build_bank, lis_dp, lis_reference
from algorithms import TrainSignals, supervision, tangent_project, train
from run_benchmark import signals_from


def test_lis_candidate_against_independent_reference_exhaustively():
    for n in range(7):
        for xs in product(range(3), repeat=n):
            assert lis_dp(xs) == lis_reference(xs)


def test_maze_honest_and_two_distinct_exploits():
    for seed in range(8):
        env = Maze(np.random.default_rng(seed))
        good = env.run((0, 0, 0))
        boundary = env.run((2, 1, 1))
        teleport = env.run((3, 1, 1))
        assert good.truth == good.proxy == good.iso == 1
        assert boundary.proxy == boundary.hard == boundary.exploit == 1
        assert boundary.truth == boundary.iso == 0
        assert teleport.proxy == teleport.causal == 1 and teleport.hard == 0
        assert teleport.truth == teleport.iso == 0


def test_grader_mutation_does_not_modify_hidden_evaluator():
    env = Grader(np.random.default_rng(4))
    assert env.run((0, 1, 0)).truth == 1
    edit = env.run((3, 2, 0))
    memorize = env.run((2, 1, 0))
    assert edit.proxy == edit.exploit == 1 and edit.truth == 0
    assert memorize.proxy == memorize.exploit == 1
    assert memorize.hard == 0 and memorize.iso == 0


def test_multistep_tampering_preserves_trusted_truth():
    env = Tamper(np.random.default_rng(4))
    honest = env.run((0, 1, 0))
    hidden = env.run((2, 1, 1))
    assert honest.truth == honest.iso == 1
    assert hidden.proxy == hidden.exploit == 1
    assert hidden.truth == hidden.iso == 0 and hidden.hard == 0


def test_training_schema_has_no_hidden_labels():
    assert "truth" not in TrainSignals.__annotations__
    assert "exploit" not in TrainSignals.__annotations__
    assert "functional" not in TrainSignals.__annotations__


def test_exact_projection_and_contract_support():
    allowed = np.array([True, False, True, True])
    g = np.array([1., 4., -3., 1.])
    p = tangent_project(g, allowed)
    assert p[1] == 0 and abs(p.sum()) < 1e-12
    assert np.allclose(tangent_project(p, allowed), p)
    sig = signals_from(build_bank("maze", 2, 11))
    _, a, _ = supervision(sig, "A_rta")
    _, d, _ = supervision(sig, "D_hierarchy")
    assert a.sum() > d.sum() > 0
    result = train(sig, "D_hierarchy", 11, updates=10)
    assert np.all(result["p"][~d] == 0)
    assert abs(result["p"].sum() - 1) < 1e-12


def test_same_seed_reproducibility():
    sig = signals_from(build_bank("tamper", 2, 31))
    a = train(sig, "B_nyaya", 7, updates=20)
    b = train(sig, "B_nyaya", 7, updates=20)
    assert np.array_equal(a["p"], b["p"])


def test_correlated_blindspot_has_false_accepts():
    m = Maze(np.random.default_rng(4), blind=True).run((3, 1, 1))
    t = Tamper(np.random.default_rng(4), blind=True).run((2, 1, 1))
    g = Grader(np.random.default_rng(4), blind=True, ood=True).run((4, 1, 0))
    for x in (m, t, g):
        assert x.hard == x.causal == 0 and x.iso == 1
        assert x.truth == 0 and x.exploit == 1


def test_A_full_and_D_are_equivalent_under_matched_information():
    sig = signals_from(build_bank("maze", 3, 81))
    a = train(sig, "A_full_constraints", 8, updates=30)
    d = train(sig, "D_hierarchy", 8, updates=30)
    assert np.allclose(a["p"], d["p"], atol=1e-12)


def test_timing_mode_does_not_change_observable_training_outputs():
    for cls in (Maze, Grader, Tamper):
        e = cls(np.random.default_rng(4))
        for plan in PLANS[::7]:
            a = e.run(plan, verify=True, evaluate=True)
            b = e.run(plan, verify=True, evaluate=False)
            for f in TrainSignals.__annotations__:
                assert getattr(a, f) == getattr(b, f)
