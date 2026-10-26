from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from phase2.trl_backend import DSLBackend, make_prompt


def test_broker_rejects_mutation_before_execution():
    backend = DSLBackend()
    a = backend.quick(make_prompt(), "0 0 0 0 1 0")
    assert not a.admissible
    assert backend.executed == 0 and backend.blocked == 1


def test_code_strings_are_not_executed():
    backend = DSLBackend()
    a = backend.quick(make_prompt(), "__import__('os').system('echo invalid')")
    assert not a.admissible and backend.executed == 0


def test_backend_audits_conditional_hack_and_accepts_legitimate_switch():
    backend = DSLBackend()
    prompt = make_prompt()
    assert backend.quick(prompt, "0 1 0 3 0 0").proxy == 1
    assert not backend.audit(prompt, "0 1 0 3 0 0")
    assert backend.audit(prompt, "0 1 0 1 0 0")


def test_challenges_and_counterfactuals_preserve_valid_task_schema():
    backend = DSLBackend()
    original = make_prompt()
    challenge = backend.challenge_prompt(original)
    paired = backend.invariant_prompt(original, "0 0 0 0 0 0")
    assert backend.parse_task(challenge)[0].suite == "grader"
    assert backend.parse_task(paired)[0].monitored is False
