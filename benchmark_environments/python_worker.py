"""Evaluate hash-verified, supplied Python functions in an isolated process.

Not a general untrusted-Python sandbox. No generated or arbitrary external file
is accepted by the public verifier. Each request is bounded by the parent timeout.
"""
import ast
import json
import sys


def validate(source):
    tree=ast.parse(source)
    permitted={"Module","FunctionDef","arguments","arg","Return","Assign","AugAssign","For","If","Expr","Call","Name","Load","Store","Constant","List","Tuple","Subscript","BinOp","UnaryOp","Compare","BoolOp","And","Or","Not","Add","Sub","Mult","Lt","LtE","Gt","GtE","Eq","NotEq","In","NotIn","USub","Attribute","IfExp"}
    if len(tree.body)!=1 or not isinstance(tree.body[0],ast.FunctionDef) or tree.body[0].name!="solve":
        raise ValueError("Expected one solve function")
    fn=tree.body[0]
    if fn.decorator_list or fn.returns or fn.args.defaults or fn.args.vararg or fn.args.kwarg or fn.args.kwonlyargs or fn.args.posonlyargs or [a.arg for a in fn.args.args]!=["x"]:
        raise ValueError("Unsupported declaration")
    if any(a.annotation for a in fn.args.args):
        raise ValueError("Annotations forbidden")
    for node in ast.walk(tree):
        if type(node).__name__ not in permitted:
            raise ValueError("Unsupported syntax")
        if isinstance(node,ast.Attribute) and (node.attr!="append" or not isinstance(node.value,ast.Name)):
            raise ValueError("Unsupported attribute access")
        if isinstance(node,ast.Call):
            valid=isinstance(node.func,ast.Name) and node.func.id in {"len","range","max","min","sum","list","enumerate","abs"}
            valid=valid or isinstance(node.func,ast.Attribute) and node.func.attr=="append"
            if not valid or node.keywords:
                raise ValueError("Unsupported call")
        if isinstance(node,ast.Name) and node.id.startswith("_"):
            raise ValueError("Unsupported name")
    return compile(tree,"<published-function>","exec")


def main():
    request=json.load(sys.stdin)
    if len(request)>500:
        raise ValueError("Evaluation batch too large")
    builtins=dict(len=len,range=range,max=max,min=min,sum=sum,list=list,enumerate=enumerate,abs=abs)
    results=[]
    for case in request:
        scope={"__builtins__":builtins}
        exec(validate(case["source"]),scope)
        budget=0
        def trace(frame,event,arg):
            nonlocal budget
            if event=="line" and frame.f_code.co_filename=="<published-function>":
                budget+=1
                if budget>150000:
                    raise RuntimeError("Instruction budget exceeded")
            return trace
        sys.settrace(trace)
        try:
            output=scope["solve"](case["input"])
        finally:
            sys.settrace(None)
        results.append(output)
    print(json.dumps(results))


if __name__=="__main__":
    main()
