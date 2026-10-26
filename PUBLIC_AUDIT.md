# Public export audit

The public directory is built from an explicit allowlist, then checked against
the private implementation before publication. The original source-bearing
research archives are **not** release assets for this repository.

## Checks performed

- No prohibited training/compiler class or function definitions in public Python.
- No imports from the private research packages or TRL; no optimizer construction,
  backward pass, or optimizer update operation in public source.
- No six-line normalized source fingerprint matching the protected training,
  adaptive-search, or compiler implementation.
- No private adaptive-arm weights, witness traces, compiler certificates, replay
  configuration, or audit-schedule configuration in exported JSON.
- All 120 neural checkpoint files contain tensor state only; no trainer or verifier state.
- File-type allowlist, credential/private-key patterns, local host-path checks,
  and repository-boundary checks.
- Seven public harness tests, including intentional hash-mismatch detection,
  independent truth grading, scale switches, masked probability consistency,
  Python permission rejection, and frozen token-table consistency.
- Live verification of all neural evaluation cells and exact integrals, and the
  optional 18-model Python evaluation. The original 46-test research suite also
  passed before export; its private training tests are not released.

The source-aware pre-publication scan reported **zero findings**. The detailed
scan implementation and protected-source fingerprints remain private to avoid
embedding the source being protected in the public audit tool.

## Limits

This is a bounded automated scan and release review, not a proof that no information
can be inferred. Public weights, final masks, token transitions, and permitted
programs intentionally reveal deployed behavior. Generic shared Python lines are
not evidence that an entire proprietary training routine was copied.

`MANIFEST.sha256` checks byte integrity of supplied files; it does not prove training
provenance or authenticity against an attacker who replaces both files and manifest.
Use the repository's release/commit identity as the distribution reference. Tests
and scripts make discrepancies visible rather than replacing live results with
saved leaderboard numbers.

Public timing tables describe reported experiments. The independent verifier
does not recreate the withheld optimizer, adaptive search, or compiler and therefore
cannot independently validate their training-cost attribution.
