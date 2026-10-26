"""Hash distributable evidence, excluding mutable validation logs and caches."""
import hashlib
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def main():
    files=[]
    for path in ROOT.rglob('*'):
        if not path.is_file():
            continue
        if any(p in {'.git','.venv','__pycache__','.pytest_cache','reproduced'} for p in path.relative_to(ROOT).parts):
            continue
        if path.name in {'SHA256SUMS.txt'} or path.suffix in {'.log','.zip'}:
            continue
        files.append(path)
    (ROOT/'SHA256SUMS.txt').write_text(''.join(f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(ROOT).as_posix()}\n'
        for p in sorted(files)),encoding='utf-8')
    print(f'Hashed {len(files)} files')


if __name__=='__main__':
    main()
