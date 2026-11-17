"""Build precomputed dataset artifacts and widget payloads for DL-4-HL.

Outputs written to ``data/artifacts/`` (and published to ``colinconwell/dl4hl``):
  - ``simple_shapes.zip`` + ``simple_shapes_meta.parquet`` (720 stratified Blender 3D shape renders)
  - ``oasis_images.zip`` + ``oasis_ratings.parquet`` (900 OASIS stimuli at 224x224 + beauty/valence/arousal ratings)
  - ``parametric_kibos.zip`` + ``parametric_kibos_meta.parquet`` (100 Kiki-Bouba stimuli from Floyd et al.)
  - ``sarcasm_headlines.parquet`` + ``sarcasm_embeddings.npz`` (28,619 headlines + 3,000 DistilBERT embeddings)
  - ``widget_samples.json`` (real-data payloads for interactive widgets)
  - ``manifest.json`` (provenance, counts, and model identifiers)
"""

from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import sys
import tarfile
import urllib.parse
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).absolute().parents[1]
RAW_CACHE = ROOT / ".cache" / "raw"
OUT_DIR = ROOT / "data" / "artifacts"

GDRIVE_SOURCES = {
    "simple_shapes.tar.bz2": "10WzhgUaRRZMQ9rm020ApZHhq6GvNLZoe",
    "sarcasm_detection.zip": "1PC1s-I34xQaiAvwUTDmA-yG2WNEzbiP9",
    "parametric_kibos.tar.gz": "1rDitrtTQeYLrREtOi-bW_H2HIT89ZowX",
    "oasis_images.tar.gz": "1qwK_yMOflGLohEKZMJwv7DqzTH_vAOc5",
    "oasis_image_ratings.csv": "1B5cj5lZf9_T1FwjqT5cxKxLbxIh-VstG",
}

ML4HL_REPO = "colinconwell/ml4hl"
ML4HL_REVISION = "a1360a0d33a7ebb77f0f14da5339cd7488e2969b"


def ensure_raw(raw_dir: Path) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    for fname, file_id in GDRIVE_SOURCES.items():
        dest = raw_dir / fname
        if dest.exists() and dest.stat().st_size > 0:
            continue
        import subprocess
        print(f"[raw] fetching {fname} ({file_id})...")
        subprocess.run(["uvx", "gdown", "-q", file_id, "-O", str(dest)], check=True)
    return raw_dir


def _save_jpeg_bytes(im: Image.Image, size: tuple[int, int] | None = None, quality: int = 86) -> bytes:
    rgb = im.convert("RGB")
    if size is not None and rgb.size != size:
        rgb = rgb.resize(size, Image.Resampling.BICUBIC)
    buf = io.BytesIO()
    rgb.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()


def build_simple_shapes(raw_dir: Path, out_dir: Path) -> dict:
    print("[1/6] Building SimpleShapes (720 stratified renders: 120 indices x 6 shapes)...")
    extract_dir = raw_dir / "_extract_shapes"
    if not (extract_dir / "SimpleShapes").exists():
        extract_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(raw_dir / "simple_shapes.tar.bz2", "r:bz2") as tf:
            tf.extractall(extract_dir)

    categories = ["Cone", "Cube", "Cylinder", "Sphere", "Suzanne", "Torus"]
    rng = np.random.default_rng(2026)
    all_indices = np.arange(1, 121)
    val_idx = set(rng.choice(all_indices, size=24, replace=False).tolist())

    rows = []
    zip_path = out_dir / "simple_shapes.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as zf:
        for idx in all_indices:
            for cat in categories:
                fname = f"{idx:05d}{cat}.jpg"
                src = extract_dir / "SimpleShapes" / fname
                with Image.open(src) as im:
                    jpg_bytes = _save_jpeg_bytes(im, size=(160, 160), quality=86)
                info = zipfile.ZipInfo(f"SimpleShapes/{fname}", date_time=(2026, 1, 1, 0, 0, 0))
                zf.writestr(info, jpg_bytes)
                rows.append({
                    "image_name": fname,
                    "image_index": int(idx),
                    "label": cat,
                    "val_set": bool(idx in val_idx),
                })

    meta = pd.DataFrame(rows).sort_values(["image_index", "label"]).reset_index(drop=True)
    meta.to_parquet(out_dir / "simple_shapes_meta.parquet", index=False)
    return {"simple_shapes_n": len(meta), "simple_shapes_classes": categories}


def build_oasis(raw_dir: Path, out_dir: Path) -> dict:
    print("[2/6] Building OASIS aesthetics dataset (900 stimuli at 224x224 + ratings)...")
    extract_dir = raw_dir / "_extract_oasis"
    oasis_src = extract_dir / "images" / "oasis"
    if not oasis_src.exists():
        extract_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(raw_dir / "oasis_images.tar.gz", "r:gz") as tf:
            tf.extractall(extract_dir)

    ratings = pd.read_csv(raw_dir / "oasis_image_ratings.csv", encoding="utf-8-sig")
    ratings["theme"] = ratings["theme"].str.strip()
    ratings = ratings.rename(columns={"category": "image_type"})
    ratings["image_name"] = ratings["theme"] + ".jpg"
    ratings = ratings.sort_values("image_name").reset_index(drop=True)

    zip_path = out_dir / "oasis_images.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as zf:
        for fname in ratings["image_name"]:
            with Image.open(oasis_src / fname) as im:
                jpg_bytes = _save_jpeg_bytes(im, size=(224, 224), quality=85)
            info = zipfile.ZipInfo(f"oasis_images/{fname}", date_time=(2026, 1, 1, 0, 0, 0))
            zf.writestr(info, jpg_bytes)

    ratings.to_parquet(out_dir / "oasis_ratings.parquet", index=False)
    return {"oasis_n": len(ratings)}


def build_kiki_bouba(raw_dir: Path, out_dir: Path) -> dict:
    print("[3/6] Building parametric Kiki-Bouba dataset (100 stimuli)...")
    extract_dir = raw_dir / "_extract_kibos"
    kibo_src = extract_dir / "parametric_kibos"
    if not kibo_src.exists():
        extract_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(raw_dir / "parametric_kibos.tar.gz", "r:gz") as tf:
            tf.extractall(extract_dir)

    rows = []
    zip_path = out_dir / "parametric_kibos.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as zf:
        for p in sorted(kibo_src.glob("*.jpg")):
            with Image.open(p) as im:
                jpg_bytes = _save_jpeg_bytes(im, size=(224, 224), quality=88)
            info = zipfile.ZipInfo(f"parametric_kibos/{p.name}", date_time=(2026, 1, 1, 0, 0, 0))
            zf.writestr(info, jpg_bytes)
            is_kiki = p.name.startswith("kiki")
            rows.append({
                "image_name": p.name,
                "shape_family": "kiki" if is_kiki else "bouba",
                "iconic_label": "kitaki" if is_kiki else "balouba",
            })

    meta = pd.DataFrame(rows)
    meta.to_parquet(out_dir / "parametric_kibos_meta.parquet", index=False)
    return {"kiki_bouba_n": len(meta)}


def build_sarcasm(raw_dir: Path, out_dir: Path) -> dict:
    print("[4/6] Building Sarcasm Headlines dataset + DistilBERT hidden-state embeddings...")
    extract_dir = raw_dir / "_extract_sarcasm"
    json_path = extract_dir / "Sarcasm_Headlines_Dataset_v2.json"
    if not json_path.exists():
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(raw_dir / "sarcasm_detection.zip") as zf:
            zf.extractall(extract_dir)

    df = pd.read_json(json_path, lines=True)
    df["source"] = df["article_link"].map(
        lambda u: urllib.parse.urlparse(str(u)).netloc.replace("www.", "").split(".")[0]
    )
    df = df.rename(columns={"headline": "sentence", "is_sarcastic": "label"})[["sentence", "label", "source"]]
    df["label"] = df["label"].astype(np.int64)
    df.to_parquet(out_dir / "sarcasm_headlines.parquet", index=False)

    # Extract 3,000 stratified DistilBERT [CLS] activations for fast SAE exploration
    import torch
    from transformers import AutoModel, AutoTokenizer

    rng = np.random.default_rng(2026)
    idx0 = rng.choice(np.where(df["label"].to_numpy() == 0)[0], size=1500, replace=False)
    idx1 = rng.choice(np.where(df["label"].to_numpy() == 1)[0], size=1500, replace=False)
    sub_idx = np.sort(np.concatenate([idx0, idx1]))
    sub_df = df.iloc[sub_idx].reset_index(drop=True)

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    model_id = "distilbert-base-uncased"
    tok = AutoTokenizer.from_pretrained(model_id)
    enc = AutoModel.from_pretrained(model_id).to(device).eval()

    embs = []
    sentences = sub_df["sentence"].tolist()
    for start in range(0, len(sentences), 128):
        batch = tok(sentences[start:start + 128], padding=True, truncation=True, max_length=64, return_tensors="pt")
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            out = enc(**batch).last_hidden_state[:, 0, :]
        embs.append(out.cpu().numpy().astype(np.float32))
    X_emb = np.concatenate(embs, axis=0)

    np.savez_compressed(
        out_dir / "sarcasm_embeddings.npz",
        embeddings=X_emb,
        sentences=np.array(sentences, dtype=object),
        labels=sub_df["label"].to_numpy(dtype=np.int64),
    )
    return {"sarcasm_n": len(df), "sarcasm_emb_n": len(sentences), "text_encoder": model_id}


def build_widget_samples(out_dir: Path) -> dict:
    print("[5/6] Precomputing real-data widget samples (Wordbank, GPT-2, SAE, OpenCLIP, DeepDive)...")
    import torch
    import open_clip
    from huggingface_hub import hf_hub_download
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression, RidgeClassifierCV, RidgeCV
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.random_projection import SparseRandomProjection
    from scipy.stats import pearsonr
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import matplotlib.cm as _cm, matplotlib.pyplot as _plt
    if not hasattr(_cm, "get_cmap"):
        _cm.get_cmap = _plt.get_cmap
    from overcomplete.sae import TopKSAE, train_sae
    from torch.utils.data import DataLoader, TensorDataset

    samples: dict = {}

    # 1. Wordbank infant vocabulary curves from colinconwell/ml4hl
    wb_path = hf_hub_download(ML4HL_REPO, "artifacts/wordbank_cdi.parquet", repo_type="dataset", revision=ML4HL_REVISION)
    wb = pd.read_parquet(wb_path).dropna(subset=["age", "vocab_share"])
    rng = np.random.default_rng(42)
    wb_sub = wb.sample(min(220, len(wb)), random_state=42).sort_values("age")
    y_col = "vocab_share"
    samples["activation_splines"] = {
        "x": np.round(wb_sub["age"].to_numpy(dtype=float), 3).tolist(),
        "y": np.round(wb_sub[y_col].to_numpy(dtype=float), 4).tolist(),
        "xlabel": "child age (months)",
        "ylabel": "vocabulary checklist proportion",
    }

    # 2. GPT-2 causal attention + 3D sphere next-token trajectories for llm_explainer
    gpt_tok = AutoTokenizer.from_pretrained("gpt2")
    gpt_mod = AutoModelForCausalLM.from_pretrained("gpt2", attn_implementation="eager").eval()
    prompts = [
        "Words become tokens that point toward",
        "In clinical interviews, anxious patients often describe",
        "Across cultures, soothing lullabies tend to sound",
        "Deep networks untangle knotted data one layer at",
    ]
    llm_cases = []
    # Fit a shared 3D PCA over GPT-2 unembedding of candidate tokens + trajectory states
    all_vecs = []
    raw_cases = []
    for text in prompts:
        inp = gpt_tok(text, return_tensors="pt")
        with torch.no_grad():
            out = gpt_mod(**inp, output_attentions=True, output_hidden_states=True)
        tok_ids = inp["input_ids"][0].tolist()
        tok_strs = [gpt_tok.decode([tid]).replace("\n", " ") for tid in tok_ids]
        logits = out.logits[0, -1, :]
        top_vals, top_ids = torch.topk(logits, 6)
        cand_strs = [gpt_tok.decode([cid]).strip() or repr(gpt_tok.decode([cid])) for cid in top_ids.tolist()]
        cand_vecs = gpt_mod.lm_head.weight[top_ids].detach().cpu().numpy()
        # Hidden states at last token across layers [0, 3, 6, 9, 12]
        layer_ids = [0, 3, 6, 9, 12]
        traj_vecs = np.stack([gpt_mod.transformer.ln_f(out.hidden_states[li][0, -1, :]).detach().cpu().numpy() if li < 12
                              else out.hidden_states[12][0, -1, :].detach().cpu().numpy()
                              for li in layer_ids])
        # Attention matrices (last 3 layers averaged vs early vs mid, head 0..3)
        attns = []
        for li in [0, 5, 11]:
            # shape: (heads, seq, seq) -> take first 4 heads
            a = out.attentions[li][0, :4].detach().cpu().numpy()
            attns.append(np.round(a, 3).tolist())
        all_vecs.append(cand_vecs)
        all_vecs.append(traj_vecs)
        raw_cases.append((text, tok_strs, tok_ids, cand_strs, top_vals.detach().cpu().numpy(), cand_vecs, traj_vecs, attns))

    V_all = np.concatenate(all_vecs, axis=0)
    V_all = V_all / (np.linalg.norm(V_all, axis=1, keepdims=True) + 1e-8)
    pca3 = PCA(n_components=3, random_state=0).fit(V_all)

    def proj_unit(mat):
        m = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8)
        p = pca3.transform(m)
        p = p / (np.linalg.norm(p, axis=1, keepdims=True) + 1e-8)
        return np.round(p, 3).tolist()

    for text, tok_strs, tok_ids, cand_strs, top_vals, cand_vecs, traj_vecs, attns in raw_cases:
        llm_cases.append({
            "prompt": text,
            "tokens": tok_strs,
            "ids": tok_ids,
            "candidates": cand_strs,
            "logits": np.round(top_vals - top_vals.max(), 3).tolist(),
            "cand_xyz": proj_unit(cand_vecs),
            "traj_xyz": proj_unit(traj_vecs),
            "attentions": attns,
        })
    samples["llm_explainer"] = {"cases": llm_cases}

    # 3. TopKSAE on Sarcasm DistilBERT embeddings for sae_superposition
    sarc = np.load(out_dir / "sarcasm_embeddings.npz", allow_pickle=True)
    X_sarc = StandardScaler().fit_transform(sarc["embeddings"].astype(np.float32))
    sents_sarc = sarc["sentences"].tolist()
    labels_sarc = sarc["labels"].tolist()
    torch.manual_seed(2026)
    sae = TopKSAE(input_shape=X_sarc.shape[1], nb_concepts=48, top_k=4, device="cpu")
    opt = torch.optim.Adam(sae.parameters(), lr=3e-3)
    loader = DataLoader(TensorDataset(torch.from_numpy(X_sarc)), batch_size=128, shuffle=True)

    def mse_crit(x, x_hat, pre_codes, codes, dictionary):
        return torch.mean((x - x_hat) ** 2)

    train_sae(sae, loader, mse_crit, opt, nb_epochs=10, monitoring=0, device="cpu")
    sae.eval()
    with torch.no_grad():
        _, codes, x_hat = sae(torch.from_numpy(X_sarc))
    codes_np = codes.numpy()
    # Select 6 well-populated, diverse SAE concepts
    firing_rate = (codes_np > 0).mean(axis=0)
    valid_dims = np.where((firing_rate > 0.03) & (firing_rate < 0.35))[0]
    peak_strength = codes_np[:, valid_dims].max(axis=0)
    chosen = valid_dims[np.argsort(peak_strength)[::-1][:6]]
    sae_features = []
    for cid in chosen:
        col = codes_np[:, cid]
        top_i = np.argsort(col)[::-1][:5]
        srate = float(np.mean([labels_sarc[i] for i in top_i]))
        sae_features.append({
            "id": int(cid),
            "firing_rate": round(float(firing_rate[cid]), 3),
            "sarcastic_share": round(srate, 2),
            "top_headlines": [
                {"text": sents_sarc[i][:88], "act": round(float(col[i]), 2), "sarcastic": int(labels_sarc[i])}
                for i in top_i
            ],
        })
    samples["sae_superposition"] = {"features": sae_features}

    # 4. OpenCLIP ViT-B-32 for Kiki-Bouba Relative Representations AND OASIS + SimpleShapes Layerwise Readouts
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    clip_model, _, clip_prep = open_clip.create_model_and_transforms("ViT-B-32-quickgelu", pretrained="openai")
    clip_tok = open_clip.get_tokenizer("ViT-B-32-quickgelu")
    clip_model = clip_model.to(device).eval()

    # 4a. Kiki-Bouba Relative Representations
    kibo_meta = pd.read_parquet(out_dir / "parametric_kibos_meta.parquet")
    kibo_imgs = []
    with zipfile.ZipFile(out_dir / "parametric_kibos.zip") as zf:
        for fname in kibo_meta["image_name"]:
            with Image.open(io.BytesIO(zf.read(f"parametric_kibos/{fname}"))) as im:
                kibo_imgs.append(clip_prep(im))
    kibo_tensor = torch.stack(kibo_imgs).to(device)
    queries = [
        "a spiky shape with sharp jagged points",
        "a smooth rounded shape with soft lobes",
        "an angular starburst or crystalline polygon",
        "a curvy amoeba or cloud silhouette",
        "a symmetrical geometric emblem",
    ]
    q_short = ["spiky / jagged", "smooth / rounded", "angular starburst", "curvy cloud", "symmetrical"]
    with torch.no_grad():
        img_emb = clip_model.encode_image(kibo_tensor)
        img_emb = (img_emb / img_emb.norm(dim=-1, keepdim=True)).cpu().numpy()
        txt_emb = clip_model.encode_text(clip_tok(queries).to(device))
        txt_emb = (txt_emb / txt_emb.norm(dim=-1, keepdim=True)).cpu().numpy()

    sim_mat = img_emb @ txt_emb.T  # (100, 5)
    y_kibo = (kibo_meta["shape_family"] == "kiki").astype(int).to_numpy()
    pca_kb = PCA(n_components=3, random_state=0).fit(np.vstack([img_emb, txt_emb]))

    def to_sphere(m):
        p = pca_kb.transform(m)
        p = p / (np.linalg.norm(p, axis=1, keepdims=True) + 1e-8)
        return np.round(p, 3).tolist()

    clf_kb = LogisticRegression(C=2.0).fit(StandardScaler().fit_transform(sim_mat), y_kibo)
    samples["relative_reps"] = {
        "queries": q_short,
        "full_queries": queries,
        "sim": np.round(sim_mat, 4).tolist(),
        "y": y_kibo.tolist(),
        "class_names": ["bouba", "kiki"],
        "img_xyz": to_sphere(img_emb),
        "query_xyz": to_sphere(txt_emb),
        "default_coefs": np.round(clf_kb.coef_[0], 3).tolist(),
    }

    # 4b. Layerwise OASIS Aesthetics Encoding + SimpleShapes Decoding
    vis_cpu = clip_model.visual.cpu().eval()
    target_blocks = [0, 2, 5, 8, 11]
    layer_labels = ["Patch stem", "Block 1", "Block 3", "Block 6", "Block 9", "Block 12", "Final norm"]

    def extract_vit_layers(pil_batch_tensors: torch.Tensor) -> dict[str, np.ndarray]:
        feats: dict[str, list[np.ndarray]] = {lbl: [] for lbl in layer_labels}
        vis_dev = clip_model.visual.to(device).eval()
        for start in range(0, len(pil_batch_tensors), 64):
            b = pil_batch_tensors[start:start + 64].to(device)
            captured: dict[str, torch.Tensor] = {}
            hooks = []
            hooks.append(vis_dev.conv1.register_forward_hook(lambda m, i, o: captured.setdefault("Patch stem", o.detach())))
            for bi, blk_idx in zip(["Block 1", "Block 3", "Block 6", "Block 9", "Block 12"], target_blocks):
                hooks.append(vis_dev.transformer.resblocks[blk_idx].register_forward_hook(
                    lambda m, i, o, name=bi: captured.setdefault(name, o.detach())
                ))
            with torch.no_grad():
                final_out = vis_dev(b)
            captured["Final norm"] = final_out.detach()
            for h in hooks:
                h.remove()
            for lbl in layer_labels:
                t = captured[lbl]
                if t.ndim == 3 and t.shape[0] != b.shape[0] and t.shape[1] == b.shape[0]:
                    t = t.permute(1, 0, 2)
                feats[lbl].append(t.reshape(b.shape[0], -1).cpu().numpy().astype(np.float32))
        return {lbl: np.concatenate(chunks, axis=0) for lbl, chunks in feats.items()}

    # OASIS (sample 360 stratified across image_type for compact widget payload; tutorial code runs full OASIS!)
    oasis_df = pd.read_parquet(out_dir / "oasis_ratings.parquet")
    oasis_sub = (
        pd.concat([g.sample(min(90, len(g)), random_state=2026) for _, g in oasis_df.groupby("image_type")])
        .sort_values("image_name")
        .reset_index(drop=True)
    )
    oasis_tensors = []
    with zipfile.ZipFile(out_dir / "oasis_images.zip") as zf:
        for fname in oasis_sub["image_name"]:
            with Image.open(io.BytesIO(zf.read(f"oasis_images/{fname}"))) as im:
                oasis_tensors.append(clip_prep(im))
    oasis_fmaps = extract_vit_layers(torch.stack(oasis_tensors))

    oasis_curves = {m: [] for m in ["beauty", "valence", "arousal"]}
    oasis_preds = {lbl: {} for lbl in layer_labels}
    oasis_rdms = {}
    rdm_idx = np.linspace(0, len(oasis_sub) - 1, 20, dtype=int)
    alphas = np.logspace(-1, 5, 19)
    for lbl in layer_labels:
        F = oasis_fmaps[lbl]
        if F.shape[1] > 512:
            F = SparseRandomProjection(n_components=512, random_state=0).fit_transform(F)
        X_sc = StandardScaler().fit_transform(F)
        rdm = 1.0 - np.corrcoef(X_sc[rdm_idx])
        oasis_rdms[lbl] = np.round(rdm, 3).tolist()
        for m in ["beauty", "valence", "arousal"]:
            y_m = oasis_sub[m].to_numpy(dtype=float)
            reg = RidgeCV(alphas=alphas, scoring="explained_variance", store_cv_results=True).fit(X_sc, y_m)
            best_i = int(np.argmin(np.abs(alphas - reg.alpha_)))
            cv_pred = reg.cv_results_[:, best_i]
            r_val = float(pearsonr(y_m, cv_pred).statistic)
            oasis_curves[m].append(round(r_val, 3))
            if m == "beauty":
                oasis_preds[lbl] = {
                    "y": np.round(y_m[::3], 2).tolist(),
                    "yhat": np.round(cv_pred[::3], 2).tolist(),
                }

    # SimpleShapes layerwise decoding curve
    shapes_df = pd.read_parquet(out_dir / "simple_shapes_meta.parquet")
    shapes_sub = shapes_df[shapes_df["image_index"] <= 50].reset_index(drop=True)  # 300 images
    shape_tensors = []
    with zipfile.ZipFile(out_dir / "simple_shapes.zip") as zf:
        for fname in shapes_sub["image_name"]:
            with Image.open(io.BytesIO(zf.read(f"SimpleShapes/{fname}"))) as im:
                shape_tensors.append(clip_prep(im))
    shape_fmaps = extract_vit_layers(torch.stack(shape_tensors))
    y_sh, sh_classes = pd.factorize(shapes_sub["label"], sort=True)
    is_val = shapes_sub["image_index"] > 38
    shape_acc = []
    for lbl in layer_labels:
        F = shape_fmaps[lbl]
        if F.shape[1] > 512:
            F = SparseRandomProjection(n_components=512, random_state=0).fit_transform(F)
        X_sc = StandardScaler().fit_transform(F)
        clf = RidgeClassifierCV(alphas=alphas).fit(X_sc[~is_val], y_sh[~is_val])
        acc = float((clf.predict(X_sc[is_val]) == y_sh[is_val]).mean())
        shape_acc.append(round(acc, 3))

    samples["layerwise_readout"] = {
        "layers": layer_labels,
        "oasis_curves": oasis_curves,
        "shapes_curve": shape_acc,
        "beauty_scatter": oasis_preds,
        "rdms": oasis_rdms,
    }

    (out_dir / "widget_samples.json").write_text(json.dumps(samples, separators=(",", ":")))
    return {"widget_samples": sorted(samples.keys()), "vision_backbone": "OpenCLIP ViT-B-32 (openai)"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-dir", type=Path, default=RAW_CACHE)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw = ensure_raw(args.raw_dir)

    meta = {"built": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "ml4hl_repo": ML4HL_REPO}
    meta.update(build_simple_shapes(raw, args.out_dir))
    meta.update(build_oasis(raw, args.out_dir))
    meta.update(build_kiki_bouba(raw, args.out_dir))
    meta.update(build_sarcasm(raw, args.out_dir))
    meta.update(build_widget_samples(args.out_dir))

    (args.out_dir / "manifest.json").write_text(json.dumps(meta, indent=2) + "\n")
    print("[6/6] Wrote manifest.json:")
    for p in sorted(args.out_dir.iterdir()):
        print(f"  {p.name:32s} {p.stat().st_size / 1e6:6.2f} MB")


if __name__ == "__main__":
    main()
