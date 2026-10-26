"""Independent checkpoint reload, final-evaluation reproduction and source audit."""
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import torch
from torch_integration import NeuralProgramPolicy
from .train import evaluate
from .shield import CompiledShield

ROOT = Path(__file__).resolve().parent


def audit_one(path):
    torch.set_num_threads(1)
    record = json.loads(Path(path).read_text())
    st = record["stats"]
    payload = torch.load(Path(path).with_suffix(".pt"), weights_only=True)
    model = NeuralProgramPolicy()
    model.load_state_dict(payload["state_dict"])
    rows = evaluate(model, st["suite"], st["method"], st["seed"], per_condition=96,
                    mask_override=CompiledShield(payload["shield"]["compiled"]))
    assert rows == record["final"], path
    return len(rows)


def main():
    meta = json.loads((ROOT / "results/metadata.json").read_text())
    for name, expected in meta["source_sha256"].items():
        assert hashlib.sha256((ROOT.parent / name).read_bytes()).hexdigest() == expected, name
    old_files = 0
    for line in (ROOT.parent / "phase2/MANIFEST.sha256").read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1)
        assert hashlib.sha256((ROOT.parent / name).read_bytes()).hexdigest() == expected, name
        old_files += 1
    paths = sorted((ROOT / "results/runs").glob("*.json"))
    with ProcessPoolExecutor(max_workers=4) as pool:
        rows = sum(pool.map(audit_one, map(str, paths)))
    result = {"status": "passed", "checkpoints_reloaded": len(paths), "rows_reproduced_exactly": rows,
              "training_source_hashes_verified": len(meta["source_sha256"]),
              "unchanged_v2_release_files_verified": old_files}
    (ROOT / "results/checkpoint_audit.json").write_text(json.dumps(result, indent=2))
    print(result)


if __name__ == "__main__":
    main()
