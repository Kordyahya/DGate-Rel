# ================================================================
# DGate-Rel paper release
# ================================================================
# This release contains only the methods reported in the manuscript:
#   1) PaperSKD: direct predecessor (dual-model WS + SKD)
#   2) DGate: reciprocal EMA peer KD + disagreement gate
#   3) DGate-Rel: DGate + relational consistency
#   4) EnsKD-DGate: disagreement-gated ensemble-target variant
#
# Controlled matrix: 3 datasets x 3 training fractions x 5 seeds.
# Seeds: 42, 43, 44, 45, 46.
# No diagnostic-only methods are exposed in this release.
# ================================================================

STAGE = "paper"
SMOKE_TEST = False

DATASETS_TO_RUN = ["COVID_Radiography", "COVID_CT", "Brain_Tumor"]
TRAIN_FRACTIONS = [0.01, 0.05, 0.10]
SEEDS = [42, 43, 44, 45, 46]
EPOCHS_BY_FRACTION = {0.01: 300, 0.05: 300, 0.10: 300}
IMPORTANCE_BATCHES = 8
PROBE_STEPS = 300
LR_GRID = None
VAL_FRACTION = 0.0

# ---------------- hyperparameters ----------------
DATA_ROOT_OVERRIDE = "/content/drive/MyDrive/EvoC_KD_2026/data"
DATASET_ROOT_OVERRIDES = {"COVID_Radiography": None, "COVID_CT": None, "Brain_Tumor": None}
BASE_SPLIT_SEED = 2026
SKIP_COMPLETED_RUNS = True
RUN_SELF_TESTS = True

TEACHER_NAME = "vit_small_patch16_224.augreg_in21k"
STUDENT_NAME = "vit_tiny_patch16_224.augreg_in21k"
IMAGE_SIZE = 224
BATCH_SIZE, MIN_BATCH = 32, 8
LEARNING_RATE, WEIGHT_DECAY, MIN_LR, LR_WARMUP_FRAC = 3e-4, 5e-2, 1e-6, 0.05
LABEL_SMOOTHING, GRAD_CLIP_NORM = 0.05, 1.0
CLASS_BALANCED_CE = True

AUGMENT, AUG_HFLIP = True, True
AUG_ROT_DEG, AUG_SCALE, AUG_SHIFT, AUG_BRIGHT, AUG_CONTRAST = 10.0, 0.10, 0.05, 0.15, 0.15

TAU, EMA_DECAY = 4.0, 0.90
LAMBDA_KD  = 0.67
LAMBDA_REL = 1.0
KD_WARMUP_EPOCHS, KD_RAMP_EPOCHS = 5, 5
DISAGREEMENT_GAMMA, GATE_CAP = 5.0, 3.0
PAPER_ALPHA = 0.6

PROBE_LR, PROBE_WD, PROBE_FOLDS = 5e-3, 1e-2, 5
IMPORTANCE_BATCH_SIZE = 16
IMPORTANCE_LAMBDA_MAG, IMPORTANCE_LAMBDA_GRAD, IMPORTANCE_LAMBDA_FISHER = 0.20, 0.40, 0.40
HIDDEN_OVERLAP = MLP_OVERLAP = LOCAL_OVERLAP = 0.10

STRICT_DATASETS = REQUIRE_COMPLETE_CLASSES = STRICT_NAME_LEAKAGE = True
STRICT_CT_GROUPS, MIN_CT_GROUP_COVERAGE, CT_MAX_UNMATCHED_IMAGES = True, 0.995, 32
CT_METADATA_FILENAME_COLUMN = CT_METADATA_GROUP_COLUMN = None

PUBLISHED_OURS = {"COVID_Radiography": {0.01: 0.750, 0.05: 0.844, 0.10: 0.869},
                  "COVID_CT":         {0.01: 0.736, 0.05: 0.828, 0.10: 0.889},
                  "Brain_Tumor":      {0.01: 0.665, 0.05: 0.774, 0.10: 0.848}}
PUBLISHED_CM1  = {"COVID_Radiography": {0.01: 0.673, 0.05: 0.791, 0.10: 0.828},
                  "COVID_CT":         {0.01: 0.694, 0.05: 0.806, 0.10: 0.870},
                  "Brain_Tumor":      {0.01: 0.663, 0.05: 0.766, 0.10: 0.839}}

# ---------------- variant definitions ----------------
def _V(name, mode, ws, target=None, dgate=False, rel=False):
    return {"name": name, "mode": mode, "ws": ws, "target": target,
            "dgate": dgate, "rel": rel}

ALL_VARIANTS = [
    _V("C1_PaperSKD",   "paper_skd", "uniform"),
    _V("DGate",         "dual_kd",   "importance", "peer", dgate=True),
    _V("DGate_Rel",     "dual_kd",   "importance", "peer", dgate=True, rel=True),
    _V("EnsKD_DGate",   "dual_kd",   "importance", "ensemble", dgate=True),
]
VARIANTS = ALL_VARIANTS.copy()

# The paper release uses the fixed learning rate reported in the manuscript.
if SMOKE_TEST:
    KD_WARMUP_EPOCHS = KD_RAMP_EPOCHS = 1

# ================== environment / install ==================
import os, sys, subprocess, importlib.util

def ensure_package(import_name, pip_spec):
    if importlib.util.find_spec(import_name) is None:
        print(f"[INSTALL] {pip_spec}")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pip_spec], check=True)

ensure_package("timm", "timm>=1.0.9")
ensure_package("sklearn", "scikit-learn>=1.5.0")
ensure_package("tqdm", "tqdm>=4.65.0")

import gc, json, math, random, time, warnings, copy, hashlib
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
from PIL import Image, ImageFile, ImageOps
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             recall_score, roc_auc_score, log_loss)
from sklearn.preprocessing import label_binarize
from tqdm.auto import tqdm
import timm

warnings.filterwarnings("ignore", category=UserWarning)
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

try:


    from google.colab import drive
    drive.mount("/content/drive", force_remount=True)

    IN_COLAB = True
except Exception:
    IN_COLAB = False
    print("[ENV] local run")

DRIVE_ROOT = "/content/drive/MyDrive" if IN_COLAB else "/content"
BASE_PROJECT_ROOT = os.path.join(DRIVE_ROOT, "EvoC_KD_2026_ComponentImpact")

# ---------------- config hash -> cache-safe directory ------------
def config_signature():
    keys = ["LEARNING_RATE","WEIGHT_DECAY","BATCH_SIZE","TAU","EMA_DECAY",
            "LAMBDA_KD","LAMBDA_REL","LABEL_SMOOTHING","GRAD_CLIP_NORM",
            "KD_WARMUP_EPOCHS","KD_RAMP_EPOCHS","PROBE_STEPS","PROBE_LR",
            "PROBE_WD","PROBE_FOLDS","IMPORTANCE_BATCHES","IMPORTANCE_BATCH_SIZE",
            "HIDDEN_OVERLAP","MLP_OVERLAP","LOCAL_OVERLAP",
            "DISAGREEMENT_GAMMA","GATE_CAP","PAPER_ALPHA",
            "AUG_ROT_DEG","AUG_SCALE","AUG_SHIFT","AUG_BRIGHT","AUG_CONTRAST",
            "CLASS_BALANCED_CE","AUGMENT","AUG_HFLIP","VAL_FRACTION"]
    g = {k: globals().get(k) for k in keys}
    g["EPOCHS_BY_FRACTION"] = EPOCHS_BY_FRACTION
    g["LR_GRID"] = LR_GRID
    return hashlib.md5(json.dumps(g, sort_keys=True, default=str).encode()).hexdigest()[:8]

RUN_SIG = config_signature()
PROJECT_ROOT = os.path.join(BASE_PROJECT_ROOT, f"{STAGE}_{RUN_SIG}")
RUN_ROOT     = os.path.join(PROJECT_ROOT, "runs")
GLOBAL_ROOT  = os.path.join(PROJECT_ROOT, "global")
MODEL_CACHE_ROOT = os.path.join(BASE_PROJECT_ROOT, "cache", "models")
LOCAL_CACHE  = "/content/evoc_v3_cache"
for p in [PROJECT_ROOT, RUN_ROOT, GLOBAL_ROOT, MODEL_CACHE_ROOT, LOCAL_CACHE]:
    os.makedirs(p, exist_ok=True)

DATA_ROOT = (os.path.abspath(os.path.expanduser(DATA_ROOT_OVERRIDE))
             if DATA_ROOT_OVERRIDE else os.path.join(PROJECT_ROOT, "data"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
AMP_ENABLED = DEVICE == "cuda"

print("=" * 78)
print(f"DGate-Rel paper release | STAGE={STAGE} | sig={RUN_SIG}")
print(f"device={DEVICE} | project={PROJECT_ROOT}")
print(f"arms={[v['name'] for v in VARIANTS]}")
print(f"datasets={DATASETS_TO_RUN} | fractions={TRAIN_FRACTIONS} | seeds={SEEDS}")
print("=" * 78)

# ==================== utilities =====================
def atomic_torch_save(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"; torch.save(obj, tmp); os.replace(tmp, path)

def atomic_json_save(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=float)
    os.replace(tmp, path)

def safe_torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        return torch.load(path, map_location="cpu", weights_only=False)

def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    except Exception:
        pass

# ==================== datasets =====================
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".ppm"}
DATASET_SPECS = {
    "COVID_Radiography": {
        "directory": "COVID-19_Radiography_Dataset",
        "classes": ["COVID", "Lung_Opacity", "Normal", "Viral_Pneumonia"],
        "folder_map": {"COVID": "COVID", "Lung_Opacity": "Lung_Opacity",
                       "Normal": "Normal", "Viral Pneumonia": "Viral_Pneumonia"}},
    "COVID_CT": {
        "directory": "COVID_CT", "classes": ["COVID", "CAP", "Normal"],
        "folder_map": {"1NonCOVID": "Normal", "2COVID": "COVID", "3CAP": "CAP"}},
    "Brain_Tumor": {
        "directory": "Brain_Tumor",
        "classes": ["glioma", "meningioma", "notumor", "pituitary"],
        "folder_map": {"glioma": "glioma", "meningioma": "meningioma",
                       "notumor": "notumor", "pituitary": "pituitary"}},
}

def norm_name(x): return "".join(ch.lower() for ch in str(x) if ch.isalnum())

def list_images(folder):
    if not os.path.isdir(folder): return []
    out = []
    with os.scandir(folder) as it:
        for e in it:
            if e.is_file() and Path(e.name).suffix.lower() in IMAGE_EXTS:
                out.append(os.path.abspath(e.path))
    return sorted(out)

def dataset_structure_ok(dataset_name, path):
    if not os.path.isdir(path): return False
    if dataset_name == "COVID_Radiography":
        return all(os.path.isdir(os.path.join(path, c, "images"))
                   for c in ["COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"])
    if dataset_name == "COVID_CT":
        return all(os.path.isdir(os.path.join(path, d))
                   for d in ["1NonCOVID", "2COVID", "3CAP"])
    if dataset_name == "Brain_Tumor":
        classes = DATASET_SPECS[dataset_name]["classes"]
        return (all(os.path.isdir(os.path.join(path, "Training", c)) for c in classes)
                and all(os.path.isdir(os.path.join(path, "Testing", c)) for c in classes))
    return False

def find_dataset_root(dataset_name):
    spec = DATASET_SPECS[dataset_name]
    override = DATASET_ROOT_OVERRIDES.get(dataset_name)
    if override:
        p = os.path.abspath(os.path.expanduser(override))
        if os.path.basename(p).lower() == spec["directory"].lower() and dataset_structure_ok(dataset_name, p):
            return p
        cand = os.path.join(p, spec["directory"])
        if dataset_structure_ok(dataset_name, cand): return cand
        raise FileNotFoundError(f"Override invalid: {override}")
    candidates = []
    for root in [DATA_ROOT, os.path.join(DRIVE_ROOT, "datasets")]:
        if not root: continue
        root = os.path.abspath(os.path.expanduser(root))
        candidates += [os.path.join(root, spec["directory"]), root]
    for c in candidates:
        if dataset_structure_ok(dataset_name, c): return os.path.abspath(c)
    if os.path.isdir(DRIVE_ROOT):
        base = DRIVE_ROOT.rstrip(os.sep).count(os.sep)
        for cur, dirs, _ in os.walk(DRIVE_ROOT):
            depth = cur.rstrip(os.sep).count(os.sep) - base
            if depth > 5: dirs.clear(); continue
            for c in [cur, os.path.join(cur, spec["directory"])]:
                if dataset_structure_ok(dataset_name, c): return os.path.abspath(c)
    return None

def index_radiography(root):
    rows = []
    for f, c in DATASET_SPECS["COVID_Radiography"]["folder_map"].items():
        files = list_images(os.path.join(root, f, "images"))
        print(f"  {c:18s}: {len(files):,}")
        rows += [{"path": p, "class_name": c, "official_split": None} for p in files]
    return rows

def index_ct(root):
    rows = []
    for f, c in DATASET_SPECS["COVID_CT"]["folder_map"].items():
        files = list_images(os.path.join(root, f))
        print(f"  {c:18s}: {len(files):,}")
        rows += [{"path": p, "class_name": c, "official_split": None} for p in files]
    return rows

def index_brain(root):
    rows = []
    for s, sl in [("Training", "train"), ("Testing", "test")]:
        for c in DATASET_SPECS["Brain_Tumor"]["classes"]:
            files = list_images(os.path.join(root, s, c))
            print(f"  {s}/{c:12s}: {len(files):,}")
            rows += [{"path": p, "class_name": c, "official_split": sl} for p in files]
    return rows

def manifest_cache_paths(dataset_name):
    out = os.path.join(GLOBAL_ROOT, dataset_name); os.makedirs(out, exist_ok=True)
    return {"manifest": os.path.join(out, "manifest.csv"),
            "split_manifest": os.path.join(out, "base_split_manifest.csv"),
            "leakage_audit": os.path.join(out, "fast_leakage_audit.json")}

def read_csv_robust(path, **kw):
    errs = []
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin1"):
        try: return pd.read_csv(path, encoding=enc, **kw), enc
        except UnicodeDecodeError as e: errs.append(f"{enc}: {e}")
    raise UnicodeError(f"{path} | " + " | ".join(errs))

def attach_labels(df, dataset_name):
    mapping = {c: i for i, c in enumerate(DATASET_SPECS[dataset_name]["classes"])}
    df = df.copy(); df["label"] = df["class_name"].map(mapping)
    if df["label"].isna().any(): raise RuntimeError("Unknown labels")
    df["label"] = df["label"].astype(int)
    return df

FILE_COL_HINTS = ["filename","file_name","file","image","image_name","imagename",
                  "image_id","imageid","path","filepath","slice","name"]
GROUP_COL_HINTS = ["patient_id","patientid","patient","case_id","caseid","case",
                   "subject_id","subjectid","subject","study_id","studyid",
                   "study","scan_id","scanid","scan"]

def column_priority(columns, hints):
    out = []
    for col in columns:
        n = norm_name(col); s = 0
        for i, h in enumerate(hints):
            nh = norm_name(h)
            if n == nh: s += 100 - i
            elif nh in n: s += 20 - min(i, 10)
        out.append((s, col))
    return [c for s, c in sorted(out, key=lambda z: (-z[0], str(z[1]))) if s > 0]

def metadata_value_variants(v):
    if pd.isna(v): return set()
    s = str(v).strip()
    if not s: return set()
    b = os.path.basename(s); st = os.path.splitext(b)[0]
    vs = {norm_name(s), norm_name(b), norm_name(st)}
    d = "".join(ch for ch in st if ch.isdigit())
    if len(d) >= 3: vs.add(d)
    return {x for x in vs if x}

def infer_metadata_columns(meta, image_paths):
    ikeys = set()
    for p in image_paths: ikeys |= metadata_value_variants(os.path.basename(p))
    fcs = column_priority(meta.columns, FILE_COL_HINTS)
    scored = []
    for c in fcs + [c for c in meta.columns if c not in fcs]:
        vals = meta[c].dropna().astype(str).head(10000)
        if not len(vals): continue
        hits = sum(1 for v in vals if metadata_value_variants(v) & ikeys)
        sc = hits / max(1, len(vals))
        sc += 0.25 * (1.0 if c in fcs[:3] else 0.0)
        scored.append((sc, c, hits, len(vals)))
    scored.sort(reverse=True)
    fc = (CT_METADATA_FILENAME_COLUMN if CT_METADATA_FILENAME_COLUMN in meta.columns
          else (scored[0][1] if scored else None))
    gcs = column_priority(meta.columns, GROUP_COL_HINTS)
    gc = CT_METADATA_GROUP_COLUMN if CT_METADATA_GROUP_COLUMN in meta.columns else None
    if gc is None:
        for c in gcs:
            if c == fc: continue
            if meta[c].notna().sum() > 0 and meta[c].nunique(dropna=True) < len(meta) * 0.99:
                gc = c; break
        if gc is None and gcs: gc = gcs[0]
    return fc, gc, scored[:10]

def match_ct_groups(df, root):
    mfs = [("COVID", os.path.join(root, "meta_data_covid.csv")),
           ("Normal", os.path.join(root, "meta_data_normal.csv")),
           ("CAP",   os.path.join(root, "meta_data_cap.csv"))]
    for _, p in mfs:
        if not os.path.isfile(p): raise FileNotFoundError(p)
    matched = pd.Series(index=df.index, dtype="object")
    source  = pd.Series(index=df.index, dtype="object")
    diags = []
    for cn, mp in mfs:
        sub = df.index[df["class_name"] == cn]
        paths = df.loc[sub, "path"].tolist()
        meta, enc = read_csv_robust(mp, dtype=str)
        fc, gc, rk = infer_metadata_columns(meta, paths)
        if fc is None or gc is None: raise RuntimeError(f"{mp} columns")
        k2g = defaultdict(set)
        for _, row in meta.iterrows():
            g = row.get(gc, None)
            if pd.isna(g) or str(g).strip() == "": continue
            g = str(g).strip()
            for k in metadata_value_variants(row.get(fc, "")): k2g[k].add(g)
        amb = {k for k, gs in k2g.items() if len(gs) > 1}
        cm = 0
        for idx in sub:
            keys = metadata_value_variants(os.path.basename(df.at[idx, "path"]))
            cand = set()
            for k in keys:
                if k in amb: continue
                cand |= k2g.get(k, set())
            if len(cand) == 1:
                matched.at[idx] = f"{cn}:{next(iter(cand))}"
                source.at[idx] = os.path.basename(mp); cm += 1
        diags.append({"class": cn, "filename_column": fc, "group_column": gc,
                      "images": len(sub), "matched": cm,
                      "coverage": cm / max(1, len(sub))})
    df = df.copy(); df["group_id"] = matched; df["group_source"] = source
    atomic_json_save(diags, os.path.join(GLOBAL_ROOT, "COVID_CT", "ct_metadata_matching.json"))
    cov = float(df["group_id"].notna().mean())
    unm = df.index[df["group_id"].isna()].to_numpy()
    print(f"[CT] coverage={cov:.4%} unmatched={len(unm)}")
    if STRICT_CT_GROUPS and len(unm) > 0:
        if cov < MIN_CT_GROUP_COVERAGE or len(unm) > CT_MAX_UNMATCHED_IMAGES:
            raise RuntimeError(f"CT unmatched {len(unm)}")
        df = df.drop(index=unm).reset_index(drop=True)
    print(f"[CT] usable {len(df):,}")
    return df

def fast_leakage_name_audit(df, tr_idx, te_idx, dataset_name):
    tr = df.iloc[np.asarray(tr_idx, dtype=int)]; te = df.iloc[np.asarray(te_idx, dtype=int)]
    exact = set(map(os.path.abspath, tr["path"])) & set(map(os.path.abspath, te["path"]))
    st = lambda p: norm_name(Path(p).stem)
    ov = set(st(p) for p in tr["path"]) & set(st(p) for p in te["path"])
    gov = set()
    if dataset_name == "COVID_CT":
        gov = set(tr["group_id"].dropna().astype(str)) & set(te["group_id"].dropna().astype(str))
    atomic_json_save({"dataset": dataset_name, "exact_path_overlap": len(exact),
                      "stem_overlap": len(ov), "group_overlap": len(gov)},
                     manifest_cache_paths(dataset_name)["leakage_audit"])
    if STRICT_NAME_LEAKAGE and (exact or ov or gov):
        raise RuntimeError(f"Leakage in {dataset_name}.")

def build_manifest(dataset_name):
    root = find_dataset_root(dataset_name)
    if root is None:
        print(f"[MISSING] {dataset_name}"); return pd.DataFrame()
    print(f"[INDEX] {dataset_name}: {root}")
    rows = {"COVID_Radiography": index_radiography, "COVID_CT": index_ct,
            "Brain_Tumor": index_brain}[dataset_name](root)
    if not rows: return pd.DataFrame()
    df = attach_labels(pd.DataFrame(rows), dataset_name)
    df["group_id"] = None; df["group_source"] = None
    if dataset_name == "COVID_CT": df = match_ct_groups(df, root)
    df = df.reset_index(drop=True)
    df["row_id"] = np.arange(len(df), dtype=np.int64)
    df.to_csv(manifest_cache_paths(dataset_name)["manifest"], index=False)
    return df

def split_by_class_groups(df, seed, test_frac=0.20):
    rng = np.random.default_rng(seed)
    groups = df["group_id"].astype(str)
    tr_g, te_g = set(), set()
    for c in sorted(df["label"].unique()):
        g = df[df["label"] == c]["group_id"].dropna().astype(str).unique().tolist()
        rng.shuffle(g)
        nt = max(1, min(len(g) - 1, int(round(len(g) * test_frac))))
        te_g.update(g[:nt]); tr_g.update(g[nt:])
    return (np.where(groups.isin(tr_g).to_numpy())[0],
            np.where(groups.isin(te_g).to_numpy())[0],
            "patient_group_stratified_80_20")

def split_dataset(df, dataset_name, split_seed):
    y = df["label"].to_numpy(); idx = np.arange(len(df))
    if dataset_name == "COVID_CT":
        tr, te, mode = split_by_class_groups(df, split_seed)
    else:
        tr, te = train_test_split(idx, test_size=0.20, random_state=split_seed, stratify=y)
        mode = "stratified_image_80_20"
    fast_leakage_name_audit(df, tr, te, dataset_name)
    return np.sort(tr), np.sort(te), mode

def load_or_make_base_split(df, dataset_name):
    path = manifest_cache_paths(dataset_name)["split_manifest"]
    if os.path.isfile(path):
        c = pd.read_csv(path)
        if {"row_id","base_split","path"}.issubset(c.columns) and len(c) == len(df) \
                and c["path"].astype(str).tolist() == df["path"].astype(str).tolist():
            tr = c.loc[c["base_split"] == "train", "row_id"].astype(int).to_numpy()
            te = c.loc[c["base_split"] == "test",  "row_id"].astype(int).to_numpy()
            if len(tr) + len(te) == len(df) and set(tr).isdisjoint(set(te)):
                fast_leakage_name_audit(df, tr, te, dataset_name)
                return np.sort(tr), np.sort(te), "cached_split"
    tr, te, mode = split_dataset(df, dataset_name, BASE_SPLIT_SEED)
    sm = df[["row_id","path","class_name","label"]].copy(); sm["base_split"] = "unused"
    sm.loc[tr, "base_split"] = "train"; sm.loc[te, "base_split"] = "test"
    sm.to_csv(path, index=False)
    return tr, te, mode

def select_fraction(df, indices, fraction, seed):
    indices = np.asarray(indices, dtype=int)
    if fraction >= 0.999999: return indices.copy()
    sub = df.iloc[indices]; rng = np.random.default_rng(seed)
    target = max(1, int(round(len(sub) * fraction)))
    labs = sub["label"].to_numpy(); u = np.unique(labs)
    cc = np.array([int(np.sum(labs == c)) for c in u])
    raw = cc / cc.sum() * target
    n = np.floor(raw).astype(int)
    rem = int(target - n.sum())
    if rem > 0:
        for i in np.argsort(-(raw - n))[:rem]: n[i] += 1
    sel = []
    for lab, k in zip(u, n):
        if k <= 0: continue
        idx = sub.index[sub["label"] == lab].to_numpy()
        sel.extend(rng.choice(idx, size=k, replace=False).tolist())
    sel = np.asarray(sel, dtype=np.int64); rng.shuffle(sel)
    print(f"[Sampling] {len(sel):,}/{len(sub):,}={100*len(sel)/len(sub):.3f}% "
          f"class_counts={dict(zip(u.tolist(), n.tolist()))}")
    return sel

def split_train_val(indices, val_fraction, seed):
    """Split fraction-selected training indices into fit/val (for LR selection)."""
    idx = np.asarray(indices)
    if val_fraction <= 0 or len(idx) < 10:
        return idx, np.array([], dtype=int)
    rng = np.random.default_rng(seed + 12345)
    perm = rng.permutation(len(idx))
    nv = max(1, int(round(len(idx) * val_fraction)))
    return idx[perm[nv:]], idx[perm[:nv]]

# ================== image cache + GPU pipeline ==================
NORM_MEAN = NORM_STD = None

def setup_norm():
    global NORM_MEAN, NORM_STD
    try:
        m = timm.create_model(STUDENT_NAME, pretrained=False, num_classes=2)
        cfg = timm.data.resolve_data_config({}, model=m)
        mean, std = cfg["mean"], cfg["std"]
    except Exception:
        mean = std = (0.5, 0.5, 0.5)
    NORM_MEAN = torch.tensor(mean, dtype=torch.float32, device=DEVICE).view(1, 3, 1, 1)
    NORM_STD  = torch.tensor(std,  dtype=torch.float32, device=DEVICE).view(1, 3, 1, 1)
    print(f"[NORM] mean={tuple(mean)} std={tuple(std)}")

def load_uint8(paths, tag):
    key = hashlib.md5("|".join(paths).encode()).hexdigest()[:10]
    cf = os.path.join(LOCAL_CACHE, f"{tag}_{key}.npy")
    if os.path.isfile(cf): return torch.from_numpy(np.load(cf))
    def _l(p):
        with Image.open(p) as im:
            im = ImageOps.exif_transpose(im).convert("RGB").resize(
                (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BICUBIC)
        return np.asarray(im, dtype=np.uint8)
    with ThreadPoolExecutor(16) as ex:
        arr = list(tqdm(ex.map(_l, paths), total=len(paths), desc=f"decode {tag}", leave=False))
    arr = np.stack(arr); np.save(cf, arr)
    return torch.from_numpy(arr)

def prep01(xu8):
    return xu8.to(DEVICE, non_blocking=True).permute(0, 3, 1, 2).float() / 255.0

def normalize(x01): return (x01 - NORM_MEAN) / NORM_STD

def gpu_augment(x):
    B, dev = x.shape[0], x.device
    r = lambda: torch.rand(B, device=dev) * 2 - 1
    ang = r() * math.radians(AUG_ROT_DEG)
    sc = 1.0 + r() * AUG_SCALE
    tx, ty = r() * AUG_SHIFT, r() * AUG_SHIFT
    fl = (torch.where(torch.rand(B, device=dev) < 0.5, -1.0, 1.0) if AUG_HFLIP
          else torch.ones(B, device=dev))
    cs, sn = torch.cos(ang) / sc, torch.sin(ang) / sc
    th = torch.stack([torch.stack([cs*fl, -sn, tx], 1),
                      torch.stack([sn*fl,  cs, ty], 1)], 1)
    grid = F.affine_grid(th, list(x.shape), align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)
    b = 1.0 + r().view(B,1,1,1) * AUG_BRIGHT
    c = 1.0 + r().view(B,1,1,1) * AUG_CONTRAST
    m = x.mean(dim=(1,2,3), keepdim=True)
    return (((x - m) * c + m) * b).clamp(0, 1)

# ====================== models =====================
def create_student(K): return timm.create_model(STUDENT_NAME, pretrained=False, num_classes=K).to(DEVICE)
def create_teacher_backbone(): return timm.create_model(TEACHER_NAME, pretrained=False, num_classes=0).to(DEVICE)

def feature_forward(model, x):
    toks = model.forward_features(x)
    if isinstance(toks, (tuple, list)): toks = toks[-1]
    logits = model.forward_head(toks, pre_logits=False)
    feat = model.forward_head(toks, pre_logits=True)
    if feat.ndim > 2: feat = feat[:, 0]
    return logits, feat

def teacher_feature_forward(teacher, x):
    toks = teacher.forward_features(x)
    if isinstance(toks, (tuple, list)): toks = toks[-1]
    feat = teacher.forward_head(toks, pre_logits=True)
    return feat[:, 0] if feat.ndim > 2 else feat

class EMA:
    def __init__(self, init_model, decay, source):
        self.decay = float(decay)
        self.model = copy.deepcopy(init_model).eval()
        for p in self.model.parameters(): p.requires_grad_(False)
        self.e = [v for v in self.model.state_dict().values() if v.is_floating_point()]
        self.s = [v for v in source.state_dict().values() if v.is_floating_point()]
    @torch.no_grad()
    def update(self):
        torch._foreach_mul_(self.e, self.decay)
        torch._foreach_add_(self.e, self.s, alpha=1.0 - self.decay)

def normalize_score(x, eps=1e-8):
    x = x.float(); fin = torch.isfinite(x)
    if not fin.any(): return torch.zeros_like(x)
    xf = x[fin]; lo, hi = xf.min(), xf.max()
    if float(hi - lo) < eps: return torch.zeros_like(x)
    out = (x - lo) / (hi - lo + eps); out[~fin] = 0
    return out.clamp(0, 1)

def load_or_download_teacher():
    cache = os.path.join(MODEL_CACHE_ROOT,
                         TEACHER_NAME.replace("/", "_").replace(":", "_") + "_backbone.pt")
    t = create_teacher_backbone()
    if os.path.isfile(cache):
        try: t.load_state_dict(safe_torch_load(cache), strict=True); return t.eval()
        except Exception: pass
    pre = timm.create_model(TEACHER_NAME, pretrained=True, num_classes=0).to(DEVICE)
    st = {k: v.detach().cpu() for k, v in pre.state_dict().items()}
    atomic_torch_save(st, cache)
    t.load_state_dict(st, strict=True)
    del pre; gc.collect()
    return t.eval()

@torch.no_grad()
def extract_feats(model, Xu8, bs=128):
    model.eval(); out = []
    for i in range(0, len(Xu8), bs):
        x = normalize(prep01(Xu8[i:i+bs]))
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=AMP_ENABLED):
            out.append(teacher_feature_forward(model, x).float().cpu())
    return torch.cat(out, 0)

# ---------- frozen-teacher linear probe (with OOF) ----------
class Probe(nn.Module):
    def __init__(self, feats, K):
        super().__init__()
        self.register_buffer("mu", feats.mean(0, keepdim=True))
        self.register_buffer("sd", feats.std(0, keepdim=True).clamp_min(1e-6))
        self.fc = nn.Linear(feats.shape[1], K)
    def forward(self, f): return self.fc((f - self.mu) / self.sd)

def fit_probe(feats, labels, K, cw, seed):
    torch.manual_seed(seed)
    p = Probe(feats, K).to(DEVICE)
    opt = torch.optim.AdamW(p.parameters(), lr=PROBE_LR, weight_decay=PROBE_WD)
    f, y = feats.to(DEVICE), labels.to(DEVICE)
    for _ in range(PROBE_STEPS):
        loss = F.cross_entropy(p(f), y, weight=cw, label_smoothing=0.1)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    return p.eval()

def oof_logits(feats, labels, K, cw, seed, return_oof_acc=False):
    n = len(labels); perm = np.random.default_rng(seed).permutation(n)
    out = torch.zeros(n, K)
    for k in range(PROBE_FOLDS):
        te = perm[k::PROBE_FOLDS]; tr = np.setdiff1d(perm, te)
        p = fit_probe(feats[tr], labels[tr], K, cw, seed + k)
        with torch.no_grad(): out[te] = p(feats[te].to(DEVICE)).cpu()
    if return_oof_acc:
        acc = float((out.argmax(1).numpy() == labels.numpy()).mean())
        return out, acc
    return out

def compute_teacher_importance(teacher, probe, Xtr_u8, ytr, seed):
    teacher.eval()
    for p in teacher.parameters(): p.requires_grad_(True)
    for p in probe.parameters():   p.requires_grad_(False)
    g = torch.Generator().manual_seed(seed + 501)
    perm = torch.randperm(len(ytr), generator=g)
    mag = {k: v.detach().abs().cpu() for k, v in teacher.named_parameters()}
    gs  = {k: torch.zeros_like(v, device="cpu") for k, v in teacher.named_parameters()}
    fs  = {k: torch.zeros_like(v, device="cpu") for k, v in teacher.named_parameters()}
    nb = 0
    for b in range(IMPORTANCE_BATCHES):
        idx = perm[b*IMPORTANCE_BATCH_SIZE:(b+1)*IMPORTANCE_BATCH_SIZE]
        if len(idx) == 0: break
        x = normalize(prep01(Xtr_u8[idx])); y = ytr[idx].to(DEVICE)
        teacher.zero_grad(set_to_none=True)
        loss = F.cross_entropy(probe(teacher_feature_forward(teacher, x)).float(), y)
        loss.backward()
        for k, p in teacher.named_parameters():
            if p.grad is None: continue
            gr = p.grad.detach().float().cpu()
            gs[k] += gr.abs(); fs[k] += gr.square()
        nb += 1
    imp = {k: (IMPORTANCE_LAMBDA_MAG  * normalize_score(mag[k])
             + IMPORTANCE_LAMBDA_GRAD * normalize_score(gs[k] / nb)
             + IMPORTANCE_LAMBDA_FISHER * normalize_score(fs[k] / nb)).float()
           for k in mag}
    for p in teacher.parameters(): p.requires_grad_(False)
    teacher.zero_grad(set_to_none=True)
    return imp

# ---------------- dual weight selection ---------------
def model_dims(t, s):
    te = int(getattr(t, "embed_dim", t.num_features))
    se = int(getattr(s, "embed_dim", s.num_features))
    return te, se, int(t.blocks[0].mlp.fc1.out_features), int(s.blocks[0].mlp.fc1.out_features)

def axis_group(key, axis, td, sd, dims):
    te, se, tm, sm = dims
    if td == sd: return "exact"
    if "qkv" in key and axis == 0 and td == 3*te and sd == 3*se: return "hidden_qkv"
    if td == te and sd == se: return "hidden"
    if td == tm and sd == sm: return "mlp"
    return "local"

def axis_importance(tensor, axis):
    if tensor.ndim == 1: return tensor.float()
    return tensor.float().mean(dim=tuple(d for d in range(tensor.ndim) if d != axis))

def uniform_pair(T, S):
    st = max(1, T // S)
    a = torch.arange(0, st*S, st)[:S]
    b = torch.clamp(a + max(1, st//2), max=T-1)
    return a, b

def select_complementary_indices(score, target_len, overlap_frac, mode="importance"):
    score = score.float().flatten()
    T, S = int(score.numel()), int(target_len)
    if T < S: raise ValueError("Teacher dim < student dim")
    if T == S:
        idx = torch.arange(S, dtype=torch.long); return idx, idx.clone(), 1.0, 0.0
    if mode == "uniform":
        a, b = uniform_pair(T, S)
        return a, b, float(len(set(a.tolist()) & set(b.tolist())))/max(1,S), 0.0
    min_ov = max(0, 2*S - T)
    ov = max(min_ov, min(S, int(round(S * overlap_frac))))
    uniq = S - ov
    try: order = torch.argsort(score, descending=True, stable=True)
    except TypeError: order = torch.argsort(score, descending=True)
    sh, rest = order[:ov], order[ov:]
    if 2*uniq > len(rest): raise RuntimeError("Complementary alloc error")
    cand = rest[:2*uniq]
    ia = torch.sort(torch.cat([sh, cand[0::2][:uniq]], 0)).values
    ib = torch.sort(torch.cat([sh, cand[1::2][:uniq]], 0)).values
    return ia, ib, float(len(set(ia.tolist()) & set(ib.tolist())))/max(1,S), 0.0

def build_group_index_maps(student, teacher, importance, mode):
    dims = model_dims(teacher, student)
    gs, lp = defaultdict(list), {}
    ss, ts = student.state_dict(), teacher.state_dict()
    for key, sv in ss.items():
        if key.startswith("head") or key not in ts: continue
        tv = ts[key]
        if tv.ndim != sv.ndim: continue
        imp = importance.get(key, tv.detach().abs().cpu())
        for ax in range(tv.ndim):
            td, sd = tv.shape[ax], sv.shape[ax]
            grp = axis_group(key, ax, td, sd, dims)
            if grp in {"hidden","mlp"}:
                gs[(grp, td, sd)].append(axis_importance(imp, ax))
            elif grp == "hidden_qkv":
                it, isd = td // 3, sd // 3
                q = imp.reshape(3, it, *imp.shape[1:])
                gs[("hidden", it, isd)].append(
                    q.mean(dim=0).mean(dim=tuple(range(1, q.ndim-1))) if q.ndim > 2 else q.mean(dim=0))
            elif grp == "local":
                lp[(key, ax)] = select_complementary_indices(axis_importance(imp, ax), sd, LOCAL_OVERLAP, mode)
    gm = {}
    for gk, sc in gs.items():
        cat = torch.stack([normalize_score(s) for s in sc]).mean(dim=0)
        gn, td, sd = gk
        gm[gk] = select_complementary_indices(cat, sd,
                  HIDDEN_OVERLAP if gn == "hidden" else MLP_OVERLAP, mode)
    return dims, gm, lp

def apply_indices(tv, idxs):
    out = tv
    for ax, idx in enumerate(idxs): out = torch.index_select(out, ax, idx.to(out.device))
    return out

def make_transfer(teacher, tmpl, importance, mode):
    ts, ss = teacher.state_dict(), tmpl.state_dict()
    dims, gm, lp = build_group_index_maps(tmpl, teacher, importance, mode)
    ma, mb = {}, {}
    for key, sv in ss.items():
        if key.startswith("head") or key not in ts: continue
        tv = ts[key].detach().cpu()
        if tv.ndim != sv.ndim: continue
        ia, ib = [], []
        for ax in range(tv.ndim):
            td, sd = tv.shape[ax], sv.shape[ax]
            grp = axis_group(key, ax, td, sd, dims)
            if grp == "exact":
                a = torch.arange(sd, dtype=torch.long); b = a.clone()
            elif grp == "hidden_qkv":
                it, isd = td // 3, sd // 3
                ha, hb, _, _ = gm[("hidden", it, isd)]
                a = torch.cat([ha + g*it for g in range(3)])
                b = torch.cat([hb + g*it for g in range(3)])
            elif grp in {"hidden","mlp"}:
                a, b, _, _ = gm[(grp, td, sd)]
            else:
                a, b, _, _ = lp[(key, ax)]
            ia.append(a); ib.append(b)
        A = apply_indices(tv, ia).float(); B = apply_indices(tv, ib).float()
        if tuple(A.shape) != tuple(sv.shape):
            raise RuntimeError(f"Shape mismatch {key}")
        ma[key], mb[key] = A, B
    if not ma: raise RuntimeError("No teacher params mapped")
    return {"a": ma, "b": mb}

def init_student(seed, K, mapped):
    seed_everything(seed)
    m = create_student(K); sd = m.state_dict()
    with torch.no_grad():
        for k, v in mapped.items(): sd[k].copy_(v.to(DEVICE, dtype=sd[k].dtype))
    return m

# ==================== losses ====================
def js_divergence(p, q, eps=1e-8):
    p, q = p.clamp_min(eps), q.clamp_min(eps); m = 0.5 * (p + q)
    return 0.5 * ((p * (p.log() - m.log())).sum(1) + (q * (q.log() - m.log())).sum(1))

def kd_loss(slog, tlog, gate, T=TAU):
    s = F.log_softmax(slog.float() / T, dim=-1)
    t = F.softmax(tlog.detach().float() / T, dim=-1)
    per = F.kl_div(s, t, reduction="none").sum(1) * (T * T)
    return per.mean() if gate is None else (gate.detach() * per).mean()

def class_norm_gate(d, y, K):
    g = torch.exp(-DISAGREEMENT_GAMMA * d)
    oh = F.one_hot(y, K).float()
    cm = (oh * g[:, None]).sum(0) / oh.sum(0).clamp_min(1.0)
    return (g / cm[y].clamp_min(1e-6)).clamp(max=GATE_CAP).detach()

def relational_loss(fa, fb):
    a, b = F.normalize(fa.float(), dim=1), F.normalize(fb.float(), dim=1)
    ga, gb = a @ a.T, b @ b.T
    diff = 0.5 * ((ga - gb.detach()).square() + (gb - ga.detach()).square())
    m = ~torch.eye(diff.shape[0], dtype=torch.bool, device=diff.device)
    return diff[m].mean()

def make_targets(kind, pa, pb):
    if kind == "peer":
        return pb, pa
    if kind == "ensemble":
        ens = 0.5 * (pa + pb)
        return ens, ens
    raise ValueError(kind)

# ==================== metrics ====================
def ece_score(y, p, n_bins=15):
    conf, pred = p.max(1), p.argmax(1); e = 0.0
    edges = np.linspace(0., 1., n_bins + 1)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i+1]
        m = (conf >= lo) & ((conf < hi) if i < n_bins-1 else (conf <= hi))
        if m.any(): e += float(m.mean()) * abs(float((pred[m]==y[m]).mean()) - float(conf[m].mean()))
    return float(e)

def basic_metrics(y, p, K):
    pred = p.argmax(1)
    r = {"accuracy": float(accuracy_score(y, pred)),
         "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
         "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
         "min_class_recall": float(recall_score(y, pred, average=None,
                                                labels=np.arange(K), zero_division=0).min()),
         "ece": ece_score(y, p),
         "brier": float(np.mean(np.sum((p - np.eye(K, dtype=np.float32)[y])**2, axis=1))),
         "nll": float(log_loss(y, p, labels=np.arange(K)))}
    try:
        r["macro_auroc"] = float(roc_auc_score(label_binarize(y, classes=np.arange(K)),
                                                p, multi_class="ovr", average="macro"))
    except Exception:
        r["macro_auroc"] = float("nan")
    return r

@torch.no_grad()
def predict_probs(model, Xte, bs=256):
    model.eval(); out = []
    for i in range(0, len(Xte), bs):
        x = normalize(prep01(Xte[i:i+bs]))
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=AMP_ENABLED):
            z = model(x)
        out.append(z.float().softmax(1).cpu())
    return torch.cat(out).numpy()

# ===================== training =====================
def lr_at(step, total, warm, base_lr):
    if step < warm: return base_lr * (step + 1) / warm
    p = (step - warm) / max(1, total - warm)
    return MIN_LR + (base_lr - MIN_LR) * 0.5 * (1 + math.cos(math.pi * p))

def run_variant(v, ctx):
    name, mode, K, seed = v["name"], v["mode"], ctx["K"], ctx["seed"]
    base_lr = v.get("lr", LEARNING_RATE)
    run_dir = os.path.join(RUN_ROOT, ctx["dataset"], name,
                           f"frac_{ctx['fraction']:.4f}".replace(".","p"), f"seed_{seed}")
    os.makedirs(run_dir, exist_ok=True)
    done = os.path.join(run_dir, "result.json")
    if SKIP_COMPLETED_RUNS and os.path.isfile(done):
        print(f"[SKIP] {name}"); return json.load(open(done))
    t0 = time.time()
    EPOCHS = ctx["epochs"]
    mapped = ctx["art"][v["ws"]]
    A = init_student(seed, K, mapped["a"])
    B = init_student(seed + 17, K, mapped["b"])
    ema_a = ema_b = aux = None
    if mode == "dual_kd": ema_a, ema_b = EMA(A, EMA_DECAY, A), EMA(B, EMA_DECAY, B)
    if mode == "paper_skd": aux = EMA(B, EMA_DECAY, A); B = None
    seed_everything(seed + 99)
    params = list(A.parameters()) + (list(B.parameters()) if B is not None else [])
    opt = torch.optim.AdamW(params, lr=base_lr, weight_decay=WEIGHT_DECAY)
    try: scaler = torch.amp.GradScaler("cuda", enabled=AMP_ENABLED)
    except Exception: scaler = torch.cuda.amp.GradScaler(enabled=AMP_ENABLED)
    crit = nn.CrossEntropyLoss(weight=ctx["cw"], label_smoothing=LABEL_SMOOTHING)
    Xtr, ytr = ctx["Xtr"], ctx["ytr"]
    n = len(ytr); spe = math.ceil(n / BATCH_SIZE); total = EPOCHS * spe
    warm = max(1, int(LR_WARMUP_FRAC * total)); step = 0
    z = torch.zeros((), device=DEVICE)
    last = {}
    for epoch in range(EPOCHS):
        g = torch.Generator().manual_seed(seed * 100003 + epoch)
        perm = torch.randperm(n, generator=g).to(DEVICE)
        acc = torch.zeros(6, device=DEVICE); nb = 0
        A.train()
        if B is not None: B.train()
        for b in range(spe):
            idx = perm[b*BATCH_SIZE:(b+1)*BATCH_SIZE]
            if len(idx) < MIN_BATCH: continue
            for pg in opt.param_groups: pg["lr"] = lr_at(step, total, warm, base_lr)
            x = prep01(Xtr[idx])
            if AUGMENT: x = gpu_augment(x)
            x = normalize(x); y = ytr[idx]
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=AMP_ENABLED):
                la, fa = feature_forward(A, x)
                ce_a = crit(la.float(), y)
                kd = rel = gm = dm = z
                if mode == "paper_skd":
                    with torch.no_grad(): lt = aux.model(x)
                    kd = kd_loss(la, lt, None)
                    loss = PAPER_ALPHA * ce_a + (1 - PAPER_ALPHA) * kd
                else:
                    lb, fb = feature_forward(B, x)
                    loss = ce_a + crit(lb.float(), y)
                    if epoch >= KD_WARMUP_EPOCHS:
                        with torch.no_grad():
                            pa = ema_a.model(x).float(); pb = ema_b.model(x).float()
                            d = js_divergence(pa.softmax(1), pb.softmax(1))
                            gate = class_norm_gate(d, y, K) if v["dgate"] else torch.ones_like(d)
                            ta, tb = make_targets(v["target"], pa, pb)
                            gm, dm = gate.mean(), d.mean()
                        kd = kd_loss(la, ta, gate) + kd_loss(lb, tb, gate)
                        ramp = min(1.0, max(0.0, (epoch+1-KD_WARMUP_EPOCHS)/max(1,KD_RAMP_EPOCHS)))
                        loss = loss + ramp * LAMBDA_KD * kd
                        if v["rel"]:
                            rel = relational_loss(fa, fb)
                            loss = loss + ramp * LAMBDA_REL * rel
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(A.parameters(), GRAD_CLIP_NORM)
            if B is not None: torch.nn.utils.clip_grad_norm_(B.parameters(), GRAD_CLIP_NORM)
            scaler.step(opt); scaler.update()
            if ema_a is not None: ema_a.update(); ema_b.update()
            if aux is not None: aux.update()
            acc += torch.stack([ce_a.detach(), kd.detach(), rel.detach(),
                                gm.detach(), dm.detach(),
                                (la.argmax(1) == y).float().mean().detach()])
            nb += 1; step += 1
        acc = (acc / max(1, nb)).tolist()
        last = {"train_ce": acc[0], "train_kd": acc[1], "train_rel": acc[2],
                "train_gate": acc[3], "train_disagree": acc[4], "train_acc_A": acc[5]}
        if (epoch+1) % 50 == 0 or epoch == 0 or epoch+1 == EPOCHS:
            print(f"  [{name}] ep {epoch+1:03d}/{EPOCHS} ce={acc[0]:.3f} kd={acc[1]:.3f} "
                  f"accA={acc[5]:.3f}")

    yte, Xte = ctx["yte"], ctx["Xte"]
    pA = predict_probs(A, Xte); mA = basic_metrics(yte, pA, K)
    row = {"dataset": ctx["dataset"], "variant": name, "fraction": ctx["fraction"],
           "seed": seed, "mode": mode, "weight_selection": v["ws"],
           "kd_target": v["target"], "dgate": int(v["dgate"]), "rel": int(v["rel"]),
           "lr_used": base_lr, "epochs": EPOCHS, "n_train": int(n), "n_test": int(len(yte)),
           **mA, **last}
    if ctx.get("has_val"):
        pV = predict_probs(A, ctx["Xval"]); mV = basic_metrics(ctx["yval"], pV, K)
        row["val_accuracy"] = mV["accuracy"]
        row["val_balanced_accuracy"] = mV["balanced_accuracy"]
        row["val_macro_f1"] = mV["macro_f1"]
    else:
        row["val_accuracy"] = float("nan")
        row["val_balanced_accuracy"] = float("nan")
        row["val_macro_f1"] = float("nan")
    nan = float("nan")
    if B is not None:
        pB = predict_probs(B, Xte); mB = basic_metrics(yte, pB, K)
        mE = basic_metrics(yte, 0.5*(pA + pB), K)
        row.update({"B_accuracy": mB["accuracy"], "B_balanced_accuracy": mB["balanced_accuracy"],
                    "B_macro_f1": mB["macro_f1"],
                    "AB_mean_balanced_accuracy": 0.5*(mA["balanced_accuracy"] + mB["balanced_accuracy"]),
                    "AB_mean_macro_f1": 0.5*(mA["macro_f1"] + mB["macro_f1"]),
                    "ens_balanced_accuracy": mE["balanced_accuracy"],
                    "ens_macro_f1": mE["macro_f1"], "ens_ece": mE["ece"]})
    else:
        row.update({"B_accuracy": nan, "B_balanced_accuracy": nan, "B_macro_f1": nan,
                    "AB_mean_balanced_accuracy": mA["balanced_accuracy"],
                    "AB_mean_macro_f1": mA["macro_f1"],
                    "ens_balanced_accuracy": nan, "ens_macro_f1": nan, "ens_ece": nan})
    row["paper_ours_acc_ref"] = PUBLISHED_OURS.get(ctx["dataset"], {}).get(ctx["fraction"], nan)
    row["paper_cm1_acc_ref"]  = PUBLISHED_CM1.get(ctx["dataset"],  {}).get(ctx["fraction"], nan)
    row["runtime_min"] = (time.time() - t0) / 60.0
    atomic_json_save(row, done)
    print(f"[DONE] {ctx['dataset']} {ctx['fraction']:.0%} {name}: "
          f"acc={row['accuracy']:.4f} bal={row['balanced_accuracy']:.4f} "
          f"f1={row['macro_f1']:.4f} ece={row['ece']:.4f} "
          f"val_acc={row['val_accuracy']:.4f} ({row['runtime_min']:.1f} min)")
    del A, B, ema_a, ema_b, aux, opt, scaler
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return row

# ============== per-setting preparation ==============
def prepare_setting(dataset, df, test_data, fraction, seed, actual_idx):
    K = len(DATASET_SPECS[dataset]["classes"])
    Xte, yte = test_data
    tag = f"{dataset}_train_{fraction}_{seed}"
    Xtr = load_uint8(df["path"].iloc[actual_idx].tolist(), tag).to(DEVICE)
    ytr = torch.tensor(df["label"].iloc[actual_idx].to_numpy(), dtype=torch.long, device=DEVICE)
    counts = torch.bincount(ytr, minlength=K).float().clamp_min(1)
    if CLASS_BALANCED_CE:
        w = counts.rsqrt(); cw = w * (len(ytr) / (w * counts).sum())
    else:
        cw = torch.ones(K, device=DEVICE)
    teacher = load_or_download_teacher()
    f_tr, f_te = extract_feats(teacher, Xtr), extract_feats(teacher, Xte)
    ytr_c = ytr.cpu()
    probe = fit_probe(f_tr, ytr_c, K, cw, seed)
    with torch.no_grad():
        ref_probs = probe(f_te.to(DEVICE)).float().softmax(1).cpu().numpy()
    ref = basic_metrics(yte, ref_probs, K)
    ref_row = {"dataset": dataset, "variant": "REF_TeacherProbe(frozen ViT-S)",
               "fraction": fraction, "seed": seed, "mode": "reference", **ref}
    print(f"[REF] frozen ViT-S probe: acc={ref['accuracy']:.4f} "
          f"bal={ref['balanced_accuracy']:.4f} f1={ref['macro_f1']:.4f} ece={ref['ece']:.4f}")
    importance = compute_teacher_importance(teacher, probe, Xtr, ytr, seed)
    tmpl = create_student(K)
    art = {m: make_transfer(teacher, tmpl, importance, m) for m in ("importance","uniform")}
    del teacher, tmpl, importance, probe, f_tr, f_te
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return {"K": K, "dataset": dataset, "fraction": fraction, "seed": seed,
            "epochs": EPOCHS_BY_FRACTION[fraction], "Xtr": Xtr, "ytr": ytr, "cw": cw,
            "Xte": Xte, "yte": yte, "art": art,
            "has_val": False}, ref_row

# ============== paired bootstrap stats ==============
def paired_bootstrap(a, b, n_boot=10000, seed=0):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    if len(a) != len(b) or len(a) == 0:
        return float("nan"), float("nan"), float("nan")
    diff = a - b
    rng = np.random.default_rng(seed)
    boots = np.array([rng.choice(diff, size=len(diff), replace=True).mean()
                      for _ in range(n_boot)])
    return (float(diff.mean()), float(np.percentile(boots, 2.5)),
            float(np.percentile(boots, 97.5)))

# ============== summary ==============
def summarize(results):
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(GLOBAL_ROOT, "all_results_v3.csv"), index=False)
    names = [v["name"] for v in VARIANTS]
    order = ["REF_TeacherProbe(frozen ViT-S)"] + names
    for frac in sorted(df["fraction"].unique()):
        g = df[df["fraction"] == frac]
        print("\n" + "="*78 + f"\nFRACTION {frac:.0%}\n" + "="*78)
        for metric in ["balanced_accuracy","macro_f1","accuracy","ece","nll",
                       "min_class_recall","AB_mean_balanced_accuracy",
                       "val_balanced_accuracy","runtime_min"]:
            if metric not in g.columns: continue
            t = g.pivot_table(index="variant", columns="dataset", values=metric, aggfunc="first")
            t = t.reindex([o for o in order if o in t.index])
            if not t.empty:
                t["MEAN"] = t.mean(axis=1)
                print(f"\n-- {metric} --"); print(t.round(4).to_string())
        if g["seed"].nunique() >= 2:
            print("\n-- paired bootstrap (95% CI) vs controls --")
            for ctrl in ["C1_PaperSKD"]:
                if ctrl not in g["variant"].unique(): continue
                for nme in names:
                    if nme == ctrl or nme not in g["variant"].unique(): continue
                    for metric in ["balanced_accuracy","macro_f1"]:
                        pa = g[g["variant"]==nme].sort_values(["dataset","seed"])[metric].values
                        pb = g[g["variant"]==ctrl].sort_values(["dataset","seed"])[metric].values
                        m, lo, hi = paired_bootstrap(pa, pb)
                        if np.isfinite(m):
                            sig = "*" if (lo > 0 or hi < 0) else " "
                            print(f"  {metric:20s} {nme:22s} vs {ctrl:12s}: "
                                  f"d={m:+.4f} [{lo:+.4f},{hi:+.4f}]{sig}")
        else:
            print("\n[single seed] paired stats require >=2 seeds — run STAGE='confirm'")
    print("\nCOMPUTE NOTE: dual_kd arms train TWO students per step (~2x C1 cost).")
    print("Reviewers will ask whether the gain justifies the cost.")
    print(f"\nAll results: {GLOBAL_ROOT}/all_results_v3.csv")

# ============== self-tests ==============
def run_self_tests():
    print("[SELF-TEST] ...")
    z = torch.randn(6, 4)
    assert torch.isfinite(kd_loss(z, z + 0.1, torch.ones(6)))
    a, b = uniform_pair(384, 192)
    assert len(a) == 192 and len(set(a.tolist()) & set(b.tolist())) == 0
    y = torch.tensor([0,0,1,1,1,2]); d = torch.rand(6) * 0.3
    g = class_norm_gate(d, y, 3)
    assert float(g.max()) >= 1.0
    x = torch.rand(4, 3, 32, 32).to(DEVICE)
    assert gpu_augment(x).shape == x.shape
    a = np.array([0.1]*10); b = np.array([0.0]*10)
    m, lo, hi = paired_bootstrap(a, b)
    assert lo > 0 and hi > 0
    print("[SELF-TEST] PASS")

# ================= main =================
def main():
    if RUN_SELF_TESTS: run_self_tests()
    setup_norm()
    all_results, t_start = [], time.time()

    n_total = len(VARIANTS) * len(SEEDS) * len(DATASETS_TO_RUN) * len(TRAIN_FRACTIONS)
    n_done = 0

    for dataset in DATASETS_TO_RUN:
        df = build_manifest(dataset)
        if df.empty:
            if STRICT_DATASETS: raise FileNotFoundError(dataset)
            continue
        missing = [c for c in DATASET_SPECS[dataset]["classes"] if (df["class_name"] == c).sum() == 0]
        if missing and REQUIRE_COMPLETE_CLASSES:
            raise RuntimeError(f"Missing classes {dataset}: {missing}")
        train_idx, test_idx, split_mode = load_or_make_base_split(df, dataset)
        print(f"[SPLIT] {dataset}: train={len(train_idx):,} test={len(test_idx):,} ({split_mode})")
        Xte = load_uint8(df["path"].iloc[test_idx].tolist(), f"{dataset}_test")
        yte = df["label"].iloc[test_idx].to_numpy()
        for fraction in TRAIN_FRACTIONS:
            for seed in SEEDS:
                full_idx = select_fraction(df, train_idx, fraction, seed + 7001)
                if VAL_FRACTION > 0:
                    fit_idx, val_idx = split_train_val(full_idx, VAL_FRACTION, seed)
                    print(f"[VAL] split: fit={len(fit_idx)} val={len(val_idx)}")
                else:
                    fit_idx, val_idx = full_idx, np.array([], dtype=int)
                ctx, ref_row = prepare_setting(dataset, df, (Xte, yte), fraction, seed, fit_idx)
                if VAL_FRACTION > 0 and len(val_idx) > 0:
                    Xv = load_uint8(df["path"].iloc[val_idx].tolist(),
                                    f"{dataset}_val_{fraction}_{seed}").to(DEVICE)
                    ctx["has_val"] = True
                    ctx["Xval"] = Xv
                    ctx["yval"] = df["label"].iloc[val_idx].to_numpy()
                all_results.append(ref_row)
                for v in VARIANTS:
                    all_results.append(run_variant(v, ctx))
                    n_done += 1
                    el = (time.time() - t_start) / 60
                    print(f"[PROGRESS] {n_done}/{n_total} | {el:.0f} min")
                    pd.DataFrame(all_results).to_csv(
                        os.path.join(GLOBAL_ROOT, "all_results_v3_partial.csv"), index=False)
                del ctx; gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()
    summarize(all_results)
    print(f"\nResults: {GLOBAL_ROOT}/all_results_v3.csv")

main()
