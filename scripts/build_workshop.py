"""Package executed workshop output and approved source downloads for Pages."""
import argparse
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).absolute().parents[1]


def build(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    text = (ROOT / "DL4HL-Tutorial.html").read_text().replace(
        'href="../ml-for-hl-part1/"', 'href="https://colinconwell.github.io/ML4HL-Workshop/"'
    )
    (out / "index.html").write_text(text)
    for name in ("DL4HL-Tutorial.qmd", "DL4HL-Tutorial.ipynb", "LICENSE", "LICENSE-HELPER"):
        if (ROOT / name).exists():
            shutil.copy2(ROOT / name, out / name)
    with zipfile.ZipFile(out / "DL4HL-Source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for folder in ("dl4hl", "assets", "scripts"):
            for p in sorted((ROOT / folder).rglob("*")):
                if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc":
                    archive.write(p, p.relative_to(ROOT))
        for name in ("DL4HL-Tutorial.qmd", "DL4HL-Tutorial.ipynb", "README.md", "pyproject.toml", "LICENSE", "LICENSE-HELPER", "JUSTFile"):
            if (ROOT / name).exists():
                archive.write(ROOT / name, name)
    (out / ".nojekyll").touch()
    print(f"Workshop site: {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "_site")
    build(ap.parse_args().out)
