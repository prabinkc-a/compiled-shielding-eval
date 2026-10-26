"""Generic witnessed-subtree backend for the open TRL adapter.

Works with a task-specific grammar exposing admissible(source). It supports
the sum and LIS tasks from the expanded discovery study. The historical maze
and size-cutoff backend remains available under phase3.python_backend.
"""
import random
from adversarial_prober import InputProber
from ast_compiler import SubtreeCompiler, run_inputs, validate


class PythonBackend:
    def __init__(self, suite, seed=0, compile_rules=True, budget=8):
        if suite not in ('tamper','grader'):
            raise ValueError('Generic backend supports sum and LIS only')
        self.suite, self.compile_rules, self.budget = suite, compile_rules, budget
        self.compiler = SubtreeCompiler(suite)
        self.prober = InputProber(seed)
        self.rules = self.compiler.rules
        self.executions = self.audits = 0

    def admissible(self, source, include_pending=True):
        try:
            tree = validate(source)
            return self.compiler.admissible(source, include_pending), tree
        except (SyntaxError, ValueError, RecursionError):
            return False, None

    def quick(self, source, step):
        if not self.admissible(source)[0]:
            return False, -3.
        rng = random.Random(600+step//8)
        x = [rng.randint(-20,20) for _ in range(8 if step%2 else 12)]
        row = run_inputs(source,self.suite,[x])[0]
        self.executions += 1
        return True, float(row['correct'])-.0001*min(row['lines'],2000)

    def audit(self, source, step):
        self.audits += 1
        for _ in range(self.budget):
            arm, x = self.prober.sample()
            row = run_inputs(source,self.suite,[x])[0]
            self.executions += 1
            self.prober.observe(arm,not row['correct'])
            if not row['correct']:
                if self.compile_rules:
                    self.compiler.propose(source,x)
                    self.executions += 1
                return False
        return True

    def publish(self):
        return self.compiler.publish()

    def state(self):
        return dict(rules=sorted(self.rules),witnesses=self.compiler.witnesses,
                    executions=self.executions,audits=self.audits)
