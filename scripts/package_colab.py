"""Package only project sources; never include personal documents or the venv."""

import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("dist/parallel-alphazero-colab.zip"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.out.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    files = [root / name for name in ("README.md", "pyproject.toml", "uv.lock", ".gitignore")]
    for folder in ("src", "configs", "tests", "docs", "scripts", "notebooks", ".github"):
        files.extend(
            p
            for p in (root / folder).rglob("*")
            if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
        )
    with ZipFile(output, "w", ZIP_DEFLATED) as archive:
        for path in files:
            if path.exists() and path.resolve() != output:
                archive.write(path, path.relative_to(root).as_posix())
    print(output)


if __name__ == "__main__":
    main()
