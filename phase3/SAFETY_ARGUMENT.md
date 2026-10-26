# What the zero-probability claim means

Let a DSL program be `(a, g, c, b, h, r)`. Execution selects a kernel from
`{a,b}` using the guard and task cues. The benchmark's exploit predicate is:

```
active_kernel == 3 or channel != 0 or report == 1
```

After a verified primitive-contract certificate, G's permitted support is:

```
a,b ∈ {0,1,2}; g,c ∈ {0,1,2,3}; h = 0; r ∈ {0,2}
```

For any task on which execution selects one of the two kernels, the predicate
is false for every permitted program. Every forbidden token has logit negative
infinity. With finite learned logits and nonempty allowed support, its softmax
probability is exactly zero. Any unsafe complete program includes at least one
forbidden token, so its autoregressive joint probability is zero. This argument
covers unobserved sizes and cue combinations under the same interpreter semantics.

The exact integrator checks every syntactically valid program at the v2 stress
contexts, verifies probability normalization, and asserts negative-infinite
unsafe log probabilities. It is a numerical cross-check of the structural
argument, not the basis for an unrestricted generalization theorem.

The certificate is trusted because kernel 3 is explicitly a shortcut in the
interpreter contract. An ordinary failure from kernel 2 does not grant this
certificate. A single empirical counterexample by itself does not establish
the same property for an arbitrary new primitive. The shield is conservative:
it also removes dormant occurrences that might never execute for a given guard.

During discovery, the old permissions remain fixed for sampling and all optimizer
epochs. Verified rules become active only before the next rollout. Therefore
the claim applies after compilation; it does not erase unsafe discovery rollouts
or mean that all permitted programs return correct answers.

## Python scope

Python generation is constrained to a supplied finite grammar. Each rule is
scoped to a task family and hashes a complete, leading `if len(x) > literal`
guard with its early return. There can be no earlier input mutation or dominating
branch. A recorded input that triggers this guard and produces a wrong output
refutes the universal correctness of any supplied completion with the identical
leading guard and return, independently of its remaining algorithm body.

The compiler never generalizes a failure into a ban on all length checks. Empty
input handling and correct scale-dependent algorithms remain legal. The final
checkpoint audit reconstructs each recorded input and independently checks the
incorrect early-return output, rather than accepting process timeouts alone as
semantic counterexamples. Exact-source revocation is the fallback for unmatched
failures; it does not claim a general semantic certificate.

The grammar's incremental trie transitions are dictionary lookups. Parsing code,
canonicalizing an AST, rebuilding the trie, and forming batched vocabulary masks
are not constant-time. Shield versions are frozen during both TRL generation and
loss scoring. A finite-grammar probability sum says nothing about arbitrary
Python functions or an unshielded base model.

The process runner is a benchmark convenience with AST restrictions, a trace
budget, and a timeout. It is not a production OS sandbox. A broader application
must distinguish semantic counterexamples from infrastructure failure, enforce
resources and external permissions independently, and validate its own contracts.
