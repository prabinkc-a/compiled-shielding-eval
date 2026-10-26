"""Create the portable v2 release, with exact hashes and prerequisite v1 data."""
from pathlib import Path
import hashlib
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
files = list(ROOT.glob("*.py")) + list(ROOT.glob("*.md")) + list(ROOT.glob("requirements*.txt"))
files += [ROOT / ".gitignore"]
files += list((ROOT / "tests").glob("*.py"))
files += [p for p in (ROOT / "results").iterdir() if p.is_file()]
for p in HERE.rglob("*"):
    if not p.is_file() or p.name == "MANIFEST.sha256":
        continue
    parts = p.relative_to(HERE).parts
    if any(x in ("pilot", "__pycache__", ".pytest_cache") for x in parts):
        continue
    files.append(p)
files = sorted(set(files))
manifest = HERE / "MANIFEST.sha256"
manifest.write_text("\n".join(
    f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(ROOT).as_posix()}"
    for p in files) + "\n", encoding="utf-8")
archive = ROOT.parent / "Frontier_AI_Alignment_Benchmark_v2.0.zip"
with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
    for p in files + [manifest]:
        z.write(p, "alignment_benchmark/" + p.relative_to(ROOT).as_posix())
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    assert not any("/.venv/" in name or "/pilot/" in name for name in z.namelist())
    for p in files:
        data = z.read("alignment_benchmark/" + p.relative_to(ROOT).as_posix())
        assert hashlib.sha256(data).hexdigest() == hashlib.sha256(p.read_bytes()).hexdigest()
print(f"Verified {len(files)+1} files in {archive} ({archive.stat().st_size:,} bytes)")
