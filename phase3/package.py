"""Create an independently hashed release without virtualenv, pilots or caches."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent.parent


def main():
    excluded = {".venv", "__pycache__", ".pytest_cache", ".git", "pilot", "pilot_python"}
    selected = []
    for path in ROOT.rglob("*"):
        rel = path.relative_to(ROOT)
        if not path.is_file() or any(p in excluded for p in rel.parts):
            continue
        if path.suffix in (".pyc", ".zip") or path.name == "MANIFEST_v3.sha256":
            continue
        # Preserve existing reports/results and dependency lock; exclude local
        # installer caches and unrelated artifacts if present in the workspace.
        if rel.parts[0] not in ("phase2", "phase3", "tests", "results") and len(rel.parts)>1:
            continue
        selected.append(path)
    manifest = ROOT / "phase3/MANIFEST_v3.sha256"
    manifest.write_text("\n".join(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(ROOT).as_posix()}"
                                   for p in sorted(selected)) + "\n", encoding="utf-8")
    selected.append(manifest)
    archive = ROOT.parent / "Frontier_AI_Alignment_Benchmark_v3.0.zip"
    with zipfile.ZipFile(archive,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in selected:
            z.write(p, "alignment_benchmark/"+p.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for line in manifest.read_text().splitlines():
            expected, name = line.split("  ",1)
            assert hashlib.sha256(z.read("alignment_benchmark/"+name)).hexdigest()==expected,name
    print(json.dumps({"archive":str(archive),"files":len(selected),"bytes":archive.stat().st_size,
                      "sha256":hashlib.sha256(archive.read_bytes()).hexdigest()},indent=2))


if __name__ == "__main__":
    main()
