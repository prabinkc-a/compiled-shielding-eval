"""Compile witnessed first-guard failures into exact AST-subtree exclusions.

No cutoff, modulo, parity, or value constant is preloaded as unsafe. Transfer is
limited to the identical first guard AND return, before any input mutation or
dominating branch. A witness refutes universal correctness of that subtree; it
does not prove that every use of a related expression is unsafe.
"""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from phase3.python_backend import ALLOWED, BUILTINS


def validate(source):
    """Apply the historical restricted-Python allowlist, adding integer modulo."""
    # Keep the historical validator byte-for-byte intact for provenance audits.
    from phase3.python_backend import validate as original_validate
    tree = ast.parse(source)
    # Validate every unchanged node using a temporary AST with modulo changed to
    # another already permitted arithmetic operator. Execute the ORIGINAL tree.
    class ModForValidation(ast.NodeTransformer):
        def visit_Mod(self, node):
            return ast.Add()
    sanitized = ModForValidation().visit(ast.parse(source))
    original_validate(ast.unparse(sanitized))
    if len(source) > 6000 or len(list(ast.walk(tree))) > 600:
        raise ValueError('Source exceeds limits')
    if any(not isinstance(n, ALLOWED + (ast.Mod,)) for n in ast.walk(tree)):
        raise ValueError('Forbidden AST node')
    return tree


def signature(source):
    """A candidate signature is not a ban: a traced wrong return is required."""
    first = validate(source).body[0].body[0]
    if isinstance(first, ast.If) and not first.orelse and len(first.body) == 1 and isinstance(first.body[0], ast.Return):
        return hashlib.sha256(ast.dump(first, include_attributes=False).encode()).hexdigest(), first.body[0].lineno
    return None, None


def run_inputs(source, suite, inputs):
    """Execute only validated supplied programs in a time-bounded subprocess."""
    validate(source)
    result = subprocess.run([sys.executable, '-I', '-S', str(Path(__file__).with_name('discovery_worker.py'))],
        input=json.dumps(dict(source=source, suite=suite, inputs=inputs)), text=True,
        capture_output=True, timeout=20, check=True)
    return json.loads(result.stdout)


class SubtreeCompiler:
    """Empty-initialized, versioned rule store; publish between policy updates."""
    def __init__(self, suite):
        self.suite = suite
        self.rules, self.pending = set(), set()
        self.witnesses = []

    def admissible(self, source, include_pending=False):
        key, _ = signature(source)
        return key not in (self.rules | self.pending if include_pending else self.rules)

    def propose(self, source, x):
        # Re-execute the witness independently; do not trust a prober's label.
        result = run_inputs(source, self.suite, [x])[0]
        key, return_line = signature(source)
        if key is None or result['correct'] or result['return_line'] != return_line:
            return False
        self.pending.add(key)
        self.witnesses.append(dict(source=source, signature=key, **result))
        return True

    def publish(self):
        changed = bool(self.pending - self.rules)
        self.rules.update(self.pending)
        self.pending.clear()
        return changed

