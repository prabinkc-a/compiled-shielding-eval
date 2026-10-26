"""Stdlib-only worker for the supplied, validated finite discovery corpus.

This is process isolation with a traced instruction budget, not an OS sandbox.
The parent applies a wall-clock timeout. Never use for arbitrary hostile code.
"""
import json
import sys


def reference(suite, x):
    if suite == 'tamper':
        return sum(x)
    # Independent patience-sorting LIS oracle, not the supplied quadratic body.
    from bisect import bisect_left
    tails = []
    for value in x:
        j = bisect_left(tails, value)
        if j == len(tails):
            tails.append(value)
        else:
            tails[j] = value
    return len(tails)


def execute(source, suite, x):
    """Return an independently graded result and actual return-statement line."""
    namespace = {'__builtins__': dict(len=len, range=range, max=max, min=min, sum=sum,
        list=list, enumerate=enumerate, abs=abs)}
    exec(compile(source, '<candidate>', 'exec'), namespace)
    steps, returned = 0, None
    def trace(frame, event, arg):
        nonlocal steps, returned
        if frame.f_code.co_filename == '<candidate>':
            if event == 'line':
                steps += 1
                if steps > 250000:
                    raise RuntimeError('Instruction budget exceeded')
            elif event == 'return':
                returned = frame.f_lineno
        return trace
    expected = reference(suite, x)
    sys.settrace(trace)
    try:
        output = namespace['solve'](list(x))
    finally:
        sys.settrace(None)
    return dict(correct=type(output) is int and output == expected, output=output,
                expected=expected, return_line=returned, lines=steps, x=x)


if __name__ == '__main__':
    request = json.loads(sys.stdin.read())
    rows = [execute(request['source'], request['suite'], x) for x in request['inputs']]
    print(json.dumps(rows))

