"""Isolated, stdlib-only process for already validated restricted benchmark code.

Not a general Python sandbox. Parent enforces AST allowlist and a 3-second kill.
An instruction budget prevents unbounded list-growing for loops in this grammar.
"""
import json
import sys


def main():
    request = json.loads(sys.stdin.read(20000))
    safe = {name: __builtins__[name] for name in ()} if isinstance(__builtins__, dict) else {}
    safe.update(len=len, range=range, max=max, min=min, sum=sum, list=list, enumerate=enumerate, abs=abs)
    namespace = {"__builtins__": safe}
    code = compile(request["source"], "<generated>", "exec")
    lines = 0
    def trace(frame, event, arg):
        nonlocal lines
        if event == "line" and frame.f_code.co_filename == "<generated>":
            lines += 1
            if lines > 150000:
                raise RuntimeError("Instruction budget exceeded")
        return trace
    exec(code, namespace)
    sys.settrace(trace)
    try:
        output = namespace["solve"](request["x"])
    finally:
        sys.settrace(None)
    print(json.dumps({"output": output, "lines": lines}))


if __name__ == "__main__":
    main()
