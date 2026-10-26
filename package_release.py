"""Package only release artifacts; exclude local environments and pilot runs."""
from pathlib import Path
import hashlib
import zipfile

ROOT = Path(__file__).resolve().parent
files = sorted(set(
    list(ROOT.glob("*.py")) + list(ROOT.glob("*.md")) + list(ROOT.glob("requirements*.txt"))
    + [ROOT / ".gitignore"] + list((ROOT / "tests").glob("*.py"))
    + [p for p in (ROOT / "results").iterdir() if p.is_file()]
))
manifest = ROOT / "MANIFEST.sha256"
manifest.write_text("\n".join(
    f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(ROOT).as_posix()}"
    for p in files) + "\n", encoding="utf-8")
archive = ROOT.parent / "Frontier_AI_Alignment_Benchmark.zip"
with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
    for p in files + [manifest]:
        z.write(p, "alignment_benchmark/" + p.relative_to(ROOT).as_posix())
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    assert not any(".venv" in n or "/pilot/" in n for n in z.namelist())
print(f"Packaged {len(files) + 1} verified files: {archive} ({archive.stat().st_size:,} bytes)")
