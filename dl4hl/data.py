"""Dataset loaders and repository helpers for DL-4-HL.

Precomputed artifacts are looked up locally first (``$DL4HL_DATA``, then
``data/artifacts`` next to the package) and otherwise fetched from the
checksum-pinned Hugging Face dataset repositories ``colinconwell/dl4hl``
(deep-learning stimuli and embeddings) and ``colinconwell/ml4hl`` (shared
domain tables).
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

PKG_ROOT = Path(__file__).absolute().parent  # not resolve(): Hub snapshots are symlinks into blobs/
CACHE = Path(os.environ.get("DL4HL_CACHE", Path.home() / ".cache" / "dl4hl"))


def _release() -> dict:
    path = PKG_ROOT / "assets" / "data-release.json"
    return json.loads(path.read_text())


def _local_dirs(artifacts=None) -> list[Path]:
    dirs: list[Path] = []
    if artifacts:
        dirs.append(Path(artifacts))
    for env_key in ("DL4HL_DATA", "ML4HL_DATA"):
        if os.environ.get(env_key):
            dirs.append(Path(os.environ[env_key]))
    dirs += [PKG_ROOT.parent / "data" / "artifacts", PKG_ROOT.parent / "artifacts"]
    return dirs


def _verify(path: Path, name: str, expected: str | None) -> Path:
    if expected and hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f"Artifact checksum differs from the pinned release: {name}")
    return path


def artifact(name: str, artifacts=None) -> Path:
    """Path to a DL4HL artifact file: local copy if present, else downloaded from the Hub."""
    rel = _release()
    expected = rel.get("files", {}).get(name)
    for d in _local_dirs(artifacts):
        if (d / name).exists():
            return _verify(d / name, name, expected)
    from huggingface_hub import hf_hub_download

    downloaded = Path(
        hf_hub_download(
            rel["repo"],
            f"artifacts/{name}",
            repo_type="dataset",
            revision=rel["revision"],
        )
    )
    return _verify(downloaded, name, expected)


def ml4hl_artifact(name: str, artifacts=None) -> Path:
    """Path to a shared ML4HL artifact file from ``colinconwell/ml4hl``."""
    rel = _release()["ml4hl"]
    expected = rel.get("files", {}).get(name)
    # Check sibling ml-for-hl-part1 artifacts if present locally
    local_candidates = _local_dirs(artifacts) + [
        PKG_ROOT.parents[1] / "ml-for-hl-part1" / "data" / "artifacts"
    ]
    for d in local_candidates:
        if (d / name).exists():
            return _verify(d / name, name, expected)
    from huggingface_hub import hf_hub_download

    downloaded = Path(
        hf_hub_download(
            rel["repo"],
            f"artifacts/{name}",
            repo_type="dataset",
            revision=rel["revision"],
        )
    )
    return _verify(downloaded, name, expected)


def artifact_manifest(artifacts=None) -> dict:
    return json.loads(artifact("manifest.json", artifacts).read_text())


def _extract_once(zip_file: Path, subdir: str, target_root: Path | None = None) -> Path:
    root = Path(target_root) if target_root else (CACHE / "extracted")
    dest = root / subdir
    sentinel = dest / ".complete"
    if sentinel.exists():
        return dest
    if dest.exists():
        shutil.rmtree(dest)
    root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_file) as zf:
        zf.extractall(root)
    sentinel.write_text(zip_file.name)
    return dest


# ─── Deep Learning Part 2 Datasets ──────────────────────────────────────


def load_simple_shapes(extract_dir=None, artifacts=None) -> dict:
    """Blender ShapeCentral 3D shape renders across 6 categories (720 images).

    Categories: ``Cone``, ``Cube``, ``Cylinder``, ``Sphere``, ``Suzanne``, ``Torus``,
    rendered under randomized position, rotation, lighting, and surface texture.
    """
    meta = pd.read_parquet(artifact("simple_shapes_meta.parquet", artifacts)).copy()
    img_dir = _extract_once(artifact("simple_shapes.zip", artifacts), "SimpleShapes", extract_dir)
    meta["image_path"] = [str(img_dir / fname) for fname in meta["image_name"]]
    classes = sorted(meta["label"].unique().tolist())
    return dict(df=meta, image_dir=img_dir, classes=classes)


def load_oasis(extract_dir=None, artifacts=None) -> dict:
    """Open Affective Standardized Image Set (OASIS, Kurdi et al., 2017; n = 900).

    Includes 224x224 RGB stimuli paired with normative human ratings of
    ``beauty``, ``valence``, and ``arousal`` across ``Person``, ``Scene``,
    ``Object``, and ``Animal`` categories.
    """
    df = pd.read_parquet(artifact("oasis_ratings.parquet", artifacts)).copy()
    img_dir = _extract_once(artifact("oasis_images.zip", artifacts), "oasis_images", extract_dir)
    df["image_path"] = [str(img_dir / fname) for fname in df["image_name"]]
    return dict(df=df, image_dir=img_dir)


def load_kiki_bouba(extract_dir=None, artifacts=None) -> dict:
    """Parametrically generated spiky (kiki) and rounded (bouba) stimuli (n = 100)."""
    df = pd.read_parquet(artifact("parametric_kibos_meta.parquet", artifacts)).copy()
    img_dir = _extract_once(artifact("parametric_kibos.zip", artifacts), "parametric_kibos", extract_dir)
    df["image_path"] = [str(img_dir / fname) for fname in df["image_name"]]
    return dict(df=df, image_dir=img_dir)


def load_sarcasm(embeddings: bool = False, artifacts=None) -> pd.DataFrame | dict:
    """News Headlines Dataset for Sarcasm Detection (Misra & Arora, 2023; n = 28,619).

    With ``embeddings=True``, also returns 3,000 stratified DistilBERT ``[CLS]``
    hidden-state vectors for fast Sparse Autoencoder (SAE) dictionary learning.
    """
    df = pd.read_parquet(artifact("sarcasm_headlines.parquet", artifacts)).copy()
    if not embeddings:
        return df
    npz = np.load(artifact("sarcasm_embeddings.npz", artifacts), allow_pickle=True)
    return dict(
        df=df,
        embeddings=npz["embeddings"].astype(np.float32),
        sentences=npz["sentences"].tolist(),
        labels=npz["labels"].astype(np.int64),
    )


DEEPDIVE_COMMIT = "e96c79a6f9c7675f85b86416c9d2b0d8462f514f"


def ensure_deepdive(dest: Path | str | None = None) -> Path:
    """Clone or download ``ColinConwell/DeepDive`` and add ``deepdive/`` to ``sys.path``."""
    root = Path(dest) if dest else (CACHE / "DeepDive")
    pkg_dir = root / "deepdive"
    if not (pkg_dir / "feature_reduction.py").exists():
        root.parent.mkdir(parents=True, exist_ok=True)
        if shutil.which("git"):
            subprocess.run(
                ["git", "clone", "-q", "--depth", "1", "https://github.com/ColinConwell/DeepDive.git", str(root)],
                check=True,
            )
        else:
            import urllib.request

            url = "https://github.com/ColinConwell/DeepDive/archive/refs/heads/main.zip"
            with urllib.request.urlopen(url, timeout=60) as resp:
                raw = resp.read()
            tmp = root.parent / "_deepdive_extract"
            if tmp.exists():
                shutil.rmtree(tmp)
            with zipfile.ZipFile(io.BytesIO(raw)) as zf:
                zf.extractall(tmp)
            extracted = next(tmp.glob("DeepDive-*"))
            if root.exists():
                shutil.rmtree(root)
            shutil.move(str(extracted), str(root))
            shutil.rmtree(tmp, ignore_errors=True)
    pkg_str = str(pkg_dir.absolute())
    if pkg_str not in sys.path:
        sys.path.insert(0, pkg_str)
    return pkg_dir


# ─── Shared ML4HL Domain Datasets ───────────────────────────────────────


def load_wordbank(language: str | None = "English (American)", artifacts=None) -> pd.DataFrame:
    """Stanford Wordbank MacArthur-Bates CDI vocabulary administrations (ages 8-30 months)."""
    df = pd.read_parquet(ml4hl_artifact("wordbank_cdi.parquet", artifacts)).copy()
    if language:
        df = df[df["language"] == language].reset_index(drop=True)
    return df


def load_perinatal_dev(artifacts=None) -> pd.DataFrame:
    """Longitudinal perinatal & child neurodevelopment cohort (IHDP, n = 837)."""
    return pd.read_parquet(ml4hl_artifact("perinatal_dev.parquet", artifacts)).copy()


def load_household_survey(artifacts=None) -> pd.DataFrame:
    """World Bank Living Standards Measurement Study (LSMS) household health & expenditure."""
    return pd.read_parquet(ml4hl_artifact("household_survey.parquet", artifacts)).copy()


def load_music_emotions(artifacts=None) -> dict:
    """Cowen et al. (2020) listener emotion ratings across 1,841 musical excerpts."""
    df = pd.read_parquet(ml4hl_artifact("music_emotions.parquet", artifacts)).copy()
    dims = [c for c in df.columns if c not in ("top_emotion", "therapy_quadrant")]
    core13 = [
        "amusing", "angry", "anxious/tense", "awe-inspiring/amazing", "calm/relaxing/serene",
        "compassionate/sympathetic", "dreamy", "eerie/mysterious", "energizing/pump-up",
        "joyful/cheerful", "sad/depressing", "scary/fearful", "triumphant/heroic",
    ]
    return dict(df=df, dimensions=dims, core13=[c for c in core13 if c in df.columns])


def load_musiclab(artifacts=None) -> dict:
    """The Music Lab Natural History of Song (NHS) & infant-directed vocalizations."""
    nhs = pd.read_parquet(ml4hl_artifact("musiclab_nhs.parquet", artifacts)).copy()
    ids = pd.read_parquet(ml4hl_artifact("musiclab_ids.parquet", artifacts)).copy()
    return dict(nhs=nhs, ids=ids)


def load_dollar_street(artifacts=None) -> dict:
    """Gapminder Dollar Street household photos, SigLIP 2 embeddings, and 48px thumbnails."""
    meta = pd.read_parquet(ml4hl_artifact("dollar_street_meta.parquet", artifacts)).copy()
    emb = np.load(ml4hl_artifact("dollar_street_emb.npy", artifacts)).astype(np.float32)
    thumbs = np.load(ml4hl_artifact("dollar_street_thumbs.npy", artifacts))
    return dict(meta=meta, emb=emb, thumbs=thumbs, model="google/siglip2-base-patch16-224")
