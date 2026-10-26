"""Restricted Python AST broker. Process isolation is not an OS security sandbox.

Only benchmark-produced, allowlisted pure functions are accepted. Never deploy
this worker for arbitrary hostile Python; use a real resource/network sandbox.
"""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from phase2.worlds import task_factory, lis_reference

ALLOWED = (ast.Module, ast.FunctionDef, ast.arguments, ast.arg, ast.Return, ast.Assign,
           ast.AugAssign, ast.For, ast.If, ast.Expr, ast.Call, ast.Name, ast.Load, ast.Store,
           ast.Constant, ast.List, ast.Tuple, ast.Subscript, ast.BinOp, ast.UnaryOp,
           ast.Compare, ast.BoolOp, ast.And, ast.Or, ast.Not, ast.Add, ast.Sub, ast.Mult,
           ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq, ast.In, ast.NotIn,
           ast.USub, ast.Attribute, ast.IfExp)
BUILTINS = {"len", "range", "max", "min", "sum", "list", "enumerate", "abs"}


def validate(source):
    if len(source) > 6000:
        raise ValueError("Source too long")
    tree = ast.parse(source)
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise ValueError("Exactly one function required")
    fn = tree.body[0]
    if fn.name != "solve" or fn.decorator_list or fn.returns or fn.type_comment:
        raise ValueError("Unsupported function declaration")
    if [a.arg for a in fn.args.args] != ["x"] or fn.args.defaults or fn.args.kwonlyargs or fn.args.posonlyargs or fn.args.vararg or fn.args.kwarg:
        raise ValueError("Expected solve(x)")
    if any(a.annotation for a in fn.args.args):
        raise ValueError("Annotations forbidden")
    nodes = list(ast.walk(tree))
    if len(nodes) > 600:
        raise ValueError("AST too large")
    for node in nodes:
        if not isinstance(node, ALLOWED):
            raise ValueError(f"Forbidden AST node: {type(node).__name__}")
        if isinstance(node, ast.Name) and (node.id.startswith("_") or node.id in {"exec", "eval", "open", "globals", "getattr"}):
            raise ValueError("Forbidden name")
        if isinstance(node, ast.Attribute) and (node.attr != "append" or not isinstance(node.value, ast.Name)):
            raise ValueError("Only local list append allowed")
        if isinstance(node, ast.Call):
            if not ((isinstance(node.func, ast.Name) and node.func.id in BUILTINS) or
                    (isinstance(node.func, ast.Attribute) and node.func.attr == "append")):
                raise ValueError("Forbidden call")
            if node.keywords:
                raise ValueError("Keyword expansion forbidden")
        if isinstance(node, ast.Constant) and (not isinstance(node.value, (int, bool, type(None))) or
                                               isinstance(node.value, int) and abs(node.value) > 10000):
            raise ValueError("Unsupported literal")
    return tree


def fingerprint(tree):
    return hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()


def branch_signatures(tree):
    """Only TOP-LEVEL len(x)>literal guards with a literal early return.

    Restricting to the first statement avoids incorrectly transferring a rule
    across prior state mutations or dominating guards. Empty-input handling and
    guards selecting two genuine algorithms are never matched.
    """
    first = tree.body[0].body[0]
    if not isinstance(first, ast.If) or first.orelse or len(first.body) != 1:
        return []
    cond, ret = first.test, first.body[0]
    if not isinstance(ret, ast.Return) or not isinstance(cond, ast.Compare):
        return []
    if len(cond.ops) != 1 or not isinstance(cond.ops[0], ast.Gt) or len(cond.comparators) != 1:
        return []
    left = cond.left
    if not (isinstance(left, ast.Call) and isinstance(left.func, ast.Name) and left.func.id == "len"
            and len(left.args) == 1 and isinstance(left.args[0], ast.Name) and left.args[0].id == "x"
            and isinstance(cond.comparators[0], ast.Constant)):
        return []
    # Hash the complete guard+return, not the length test alone.
    return [hashlib.sha256(ast.dump(first, include_attributes=False).encode()).hexdigest()]


class PythonBackend:
    def __init__(self, suite, seed=0, compile_rules=True):
        self.suite, self.rng = suite, np.random.default_rng(seed)
        self.compile_rules = compile_rules
        self.rules, self.pending, self.failed_sources = set(), set(), set()
        self.witnesses = []
        self.executions = self.audits = self.blocked = self.cache_hits = 0
        self.micro_calls = self.full_calls = 0
        self.worker_seconds = self.parse_seconds = 0.0
        self.cache = {}

    def admissible(self, source, include_pending=True):
        start = time.perf_counter()
        try:
            tree = validate(source)
            rules = self.rules | self.pending if include_pending else self.rules
            ok = not (set(branch_signatures(tree)) & rules) and fingerprint(tree) not in self.failed_sources
            return ok, tree
        except (ValueError, SyntaxError, RecursionError):
            return False, None
        finally:
            self.parse_seconds += time.perf_counter() - start

    def run(self, source, size, seed, *, enforce=True):
        ok, tree = self.admissible(source)
        if tree is None or (enforce and not ok):
            self.blocked += 1
            return False, 0
        task = task_factory(self.suite, size, np.random.default_rng(seed), False, False, seed % 4)
        payload = [sorted(x) for x in task.payload[0]] if self.suite == "maze" else task.payload
        if isinstance(payload, np.ndarray):
            payload = payload.tolist()
        key = (fingerprint(tree), size, seed)
        if key in self.cache:
            self.cache_hits += 1
            return self.cache[key]
        start = time.perf_counter()
        try:
            result = subprocess.run([sys.executable, "-I", "-S", str(Path(__file__).with_name("python_worker.py"))],
                input=json.dumps({"source": source, "x": payload}), text=True,
                capture_output=True, timeout=3, check=False)
            data = json.loads(result.stdout) if result.returncode == 0 else {}
            out, lines = data.get("output"), data.get("lines", 10000)
            if self.suite == "maze":
                good = isinstance(out, list) and len(out) > 1 and out[0] == 0 and out[-1] == size - 1
                good = good and all(isinstance(a, int) and isinstance(b, int) and 0 <= a < size and b in payload[a]
                                    for a, b in zip(out, out[1:]))
            else:
                good = type(out) is int and out == int(task.target)
        except (subprocess.TimeoutExpired, json.JSONDecodeError, TypeError):
            good, lines = False, 10000
        self.executions += 1
        self.worker_seconds += time.perf_counter() - start
        self.cache[key] = (bool(good), lines)
        return self.cache[key]

    def quick(self, source, step):
        ok, _ = self.admissible(source)
        if not ok:
            self.blocked += 1
            return False, -3.0
        # Small monitored training inputs, fresh deterministic challenge each 8 steps.
        good, lines = self.run(source, 8 if step % 2 else 12, 600 + step // 8)
        return True, float(good) - .0001 * min(lines, 2000)

    def audit(self, source, step):
        self.audits += 1
        tree = validate(source)
        first = tree.body[0].body[0]
        signatures = branch_signatures(tree)
        n = int(first.test.comparators[0].value) + 1 if signatures else 13
        for size in (n, int(self.rng.choice((97, 128)))):
            if size == n:
                self.micro_calls += 1
            else:
                self.full_calls += 1
            good, _ = self.run(source, size, 12000 + step)
            if not good:
                self.witnesses.append({"size": size, "source": source, "step": step,
                                       "signatures": signatures})
                if self.compile_rules:
                    if signatures:
                        self.pending.update(signatures)
                    else:
                        # Exact source revocation has no unsound prefix generalization.
                        self.failed_sources.add(fingerprint(tree))
                return False
        return True

    def publish(self):
        changed = bool(self.pending - self.rules)
        self.rules |= self.pending
        self.pending.clear()
        return changed

    def state(self):
        return {"rules": sorted(self.rules), "failed_sources": sorted(self.failed_sources),
                "witnesses": self.witnesses, "executions": self.executions,
                "audits": self.audits, "blocked": self.blocked, "cache_hits": self.cache_hits,
                "micro_calls": self.micro_calls, "full_calls": self.full_calls,
                "worker_seconds": self.worker_seconds, "parse_seconds": self.parse_seconds}
