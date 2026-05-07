"""
Decision Voting Based Multiscale Graph Convolutional Network
for ADHD Classification on the ADHD-200 Preprocessed Anatomical Dataset

FIXES in this version:
  1. load_labels() now ALSO scans the BASE_PATH root for site-named phenotypic
     CSVs (e.g. KKI_phenotypic.csv, NYU_phenotypic.csv) which is where the
     actual diagnosis labels live in this Kaggle dataset layout.
  2. The root-level CSV subject-ID column is "ScanDir ID" (with a space) —
     handled explicitly with a fallback chain.
  3. The diagnosis column in root CSVs uses numeric codes: 0=TD, 1/2/3=ADHD.
  4. Subject-ID normalisation is consistent between label loading and NIfTI
     discovery (both strip "sub-" prefix and leading zeros are NOT stripped).
  5. Peking_3 site has only 128-scale NIfTIs (no 224 filtered_participants.tsv
     with 224 paths) — subjects missing either scale are silently dropped as
     before via the intersection logic.
"""

# ═══════════════════════════════════════════════════════════════════
# IMPORTS
# ═══════════════════════════════════════════════════════════════════
import os
import re
import warnings
import numpy as np
import pandas as pd
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats as scipy_stats

from torch_geometric.nn import ChebConv, global_mean_pool
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    matthews_corrcoef, recall_score, fowlkes_mallows_score,
    confusion_matrix, precision_score,
)
warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════
BASE_PATH = (
    "/kaggle/input/datasets/purnimakumarrr/"
    "adhd200-preprocessed-anatomical-dataset/adhd200-preprocessed"
)

SITES = [
    "KKI", "NYU", "OHSU",
    "Peking_1", "Peking_2", "Peking_3",
    "Pittsburgh", "WashU", "NeuroIMAGE",
]

SCALE_CONFIGS = {
    128: {"grid": (4, 4, 4),  "n_nodes": 64},
    224: {"grid": (8, 8, 8),  "n_nodes": 512},
}
N_FEATURES  = 8    # mean, std, min, max, q25, q75, skew, kurtosis

K_CHEB       = 3
HIDDEN       = 64
DROPOUT      = 0.5
LR           = 0.01
WEIGHT_DECAY = 1e-4
BATCH_SIZE   = 32
EPOCHS       = 100
N_FOLDS      = 5

SIGMA_NOISE  = 2.0
N_AUGMENTS   = 4

SEED   = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED)
np.random.seed(SEED)

print(f"Device  : {DEVICE}")
print(f"Scales  : {list(SCALE_CONFIGS.keys())}")
print(f"Nodes   : {[v['n_nodes'] for v in SCALE_CONFIGS.values()]}")
print(f"Features: {N_FEATURES} per node")
print(f"Aug ×   : {N_AUGMENTS + 1}  |  K={K_CHEB}  |  σ={SIGMA_NOISE}")
print()


# ═══════════════════════════════════════════════════════════════════
# 1.  DATA DISCOVERY AND LABEL LOADING
# ═══════════════════════════════════════════════════════════════════

def _parse_label(val):
    """Convert raw phenotypic value to 0 (TD) or 1 (ADHD)."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    s = str(val).strip().lower()
    if s in ("0", "td", "control", "typically developing", "healthy",
             "typically_developing"):
        return 0
    if s in ("adhd", "1", "2", "3", "adhd-c", "adhd-i", "adhd-ph"):
        return 1
    try:
        v = int(float(s))
        if v == 0:
            return 0
        if v in (1, 2, 3):
            return 1
    except (ValueError, OverflowError):
        pass
    return None


def _normalise_subject_id(raw: str) -> str:
    """Strip leading 'sub-' prefix (case-insensitive) and whitespace."""
    s = str(raw).strip()
    s = re.sub(r"(?i)^sub[-_]?", "", s)
    return s.strip()


def _extract_labels_from_df(df: pd.DataFrame, source_tag: str) -> dict:
    """
    Try many column-name variants to pull (subject_id -> 0/1) from a
    DataFrame.  Returns {} on failure.
    """
    # Priority-ordered candidate column names
    SUB_COLS = [
        "participant_id", "Participant_ID",
        "ScanDir ID",     # root-level ADHD-200 phenotypic CSVs
        "ScanDir_ID",
        "subject", "Subject",
        "ID", "id",
        "subjectkey", "SubjectKey", "SUBJECTKEY",
    ]
    DX_COLS = [
        "dx_group", "DX_Group", "DX_GROUP", "DX",
        "ADHD", "adhd",
        "Diagnosis", "diagnosis",
        "Group", "group",
        "dx",
    ]

    # Find first matching column
    sub_col = next((c for c in SUB_COLS if c in df.columns), None)
    dx_col  = next((c for c in DX_COLS  if c in df.columns), None)

    # Fallback: substring search
    if sub_col is None or dx_col is None:
        for col in df.columns:
            cl = col.lower()
            if sub_col is None and any(k in cl for k in
                                       ("subject", "participant", "scandir", "id")):
                sub_col = col
            if dx_col is None and any(k in cl for k in
                                      ("dx", "diag", "group", "adhd")):
                dx_col = col

    if sub_col is None or dx_col is None:
        print(f"    [WARN] {source_tag}: cannot identify sub/dx columns. "
              f"Columns = {list(df.columns)}")
        return {}

    out = {}
    for _, row in df.iterrows():
        sid   = _normalise_subject_id(row[sub_col])
        label = _parse_label(row[dx_col])
        if sid and label is not None:
            out[sid] = label

    print(f"  {source_tag}: loaded {len(out)} labels  "
          f"[sub_col='{sub_col}', dx_col='{dx_col}']")
    return out


def load_labels(base_path: str, sites: list) -> dict:
    """
    Two-pass label loading:

    Pass 1 — ROOT-LEVEL phenotypic CSVs
        Pattern: <BASE_PATH>/<SITE>_phenotypic.csv  (and *_TestRelease_*)
        These are the primary label source for this Kaggle dataset.

    Pass 2 — SITE-LEVEL TSV/CSV files
        <BASE_PATH>/<SITE>/filtered_participants.tsv  (or participants.tsv)
        Used as fallback when the root CSV has no labels for that site.
    """
    label_map: dict = {}

    # ── Pass 1: root-level phenotypic CSVs ──────────────────────────
    print("  [Pass 1] Scanning root-level phenotypic CSVs …")
    for fname in os.listdir(base_path):
        if not fname.endswith(".csv"):
            continue
        # Match files like KKI_phenotypic.csv, OHSU_TestRelease_phenotypic.csv
        site_match = None
        for site in sites:
            if fname.upper().startswith(site.upper()):
                site_match = site
                break
        if site_match is None:
            continue

        fpath = os.path.join(base_path, fname)
        try:
            df  = pd.read_csv(fpath, low_memory=False, dtype=str)
            got = _extract_labels_from_df(df, f"ROOT/{fname}")
            label_map.update(got)
        except Exception as exc:
            print(f"    [WARN] ROOT/{fname}: read error – {exc}")

    print(f"  [Pass 1 total] {len(label_map)} labels from root CSVs")

    # ── Pass 2: site-level TSV / CSV (fallback) ──────────────────────
    print("  [Pass 2] Scanning site-level TSV/CSV files …")
    candidates = ["filtered_participants.tsv", "participants.tsv"]

    for site in sites:
        sp = os.path.join(base_path, site)
        if not os.path.isdir(sp):
            print(f"  [WARN] Site directory not found: {sp}")
            continue

        found = False
        for fname in candidates:
            fpath = os.path.join(sp, fname)
            if not os.path.isfile(fpath):
                continue
            try:
                df  = pd.read_csv(fpath, sep="\t", low_memory=False, dtype=str)
                got = _extract_labels_from_df(df, f"{site}/{fname}")
                # Only add IDs not already found in pass 1
                new = {k: v for k, v in got.items() if k not in label_map}
                label_map.update(new)
                found = True
                break
            except Exception as exc:
                print(f"  [WARN] {site}/{fname}: read error – {exc}")

        if not found:
            # Try any CSV in the site directory
            for fname in os.listdir(sp):
                if not fname.endswith(".csv"):
                    continue
                fpath = os.path.join(sp, fname)
                try:
                    df  = pd.read_csv(fpath, low_memory=False, dtype=str)
                    got = _extract_labels_from_df(df, f"{site}/{fname}")
                    new = {k: v for k, v in got.items() if k not in label_map}
                    label_map.update(new)
                    break
                except Exception:
                    pass

    return label_map


def find_nifti_files(base_path: str, sites: list, scale: int) -> dict:
    """
    Returns {normalised_subject_id → absolute_path} for NIfTI files at
    the given scale.  Matches files whose names start with
    normalized_resampled_{scale}_ and end with .nii.
    """
    prefix  = f"normalized_resampled_{scale}_"
    fmap: dict = {}

    for site in sites:
        sp = os.path.join(base_path, site)
        if not os.path.isdir(sp):
            continue
        for entry in os.scandir(sp):
            if not entry.is_dir():
                continue
            if not entry.name.lower().startswith("sub-"):
                continue
            sid = _normalise_subject_id(entry.name)

            for fname in os.listdir(entry.path):
                if fname.startswith(prefix) and fname.endswith(".nii"):
                    fmap[sid] = os.path.join(entry.path, fname)
                    break

    return fmap


def build_dataset(base_path: str, sites: list):
    """Intersect labels + both scales → list of subject dicts + label array."""
    if not os.path.isdir(base_path):
        raise FileNotFoundError(
            f"Dataset root not found: {base_path}\n"
            "Please update BASE_PATH."
        )

    print("Loading phenotypic labels …")
    labels = load_labels(base_path, sites)
    print(f"  Total labels found : {len(labels)}")

    if not labels:
        raise RuntimeError(
            "No labels were loaded. Check phenotypic files."
        )

    print("Discovering NIfTI files …")
    f128 = find_nifti_files(base_path, sites, 128)
    f224 = find_nifti_files(base_path, sites, 224)
    print(f"  Scale-128 : {len(f128)} files")
    print(f"  Scale-224 : {len(f224)} files")

    lbl_ids  = set(labels.keys())
    f128_ids = set(f128.keys())
    f224_ids = set(f224.keys())

    only_lbl = lbl_ids - f128_ids - f224_ids
    only_nii = (f128_ids | f224_ids) - lbl_ids
    if only_lbl:
        print(f"  [DEBUG] In labels but not NIfTI (first 5): "
              f"{list(only_lbl)[:5]}")
    if only_nii:
        print(f"  [DEBUG] In NIfTI but not labels (first 5): "
              f"{list(only_nii)[:5]}")

    common = sorted(lbl_ids & f128_ids & f224_ids)
    print(f"  Valid subjects (both scales + label): {len(common)}")

    if len(common) == 0:
        raise RuntimeError(
            "No subjects matched across labels AND both NIfTI scales.\n"
            "Possible causes:\n"
            "  • Subject IDs in phenotypic files differ from directory names.\n"
            "  • NIfTI files for one scale are missing.\n"
            "  • BASE_PATH is wrong."
        )

    dataset = [
        {"sub_id": s, "path_128": f128[s], "path_224": f224[s],
         "label": labels[s]}
        for s in common
    ]
    y = np.array([d["label"] for d in dataset])
    print(f"  TD={int((y==0).sum())}  ADHD={int((y==1).sum())}")
    return dataset, y


# ═══════════════════════════════════════════════════════════════════
# 2.  VOLUME → PATCH-GRAPH CONVERSION
# ═══════════════════════════════════════════════════════════════════

def load_volume(path: str) -> np.ndarray:
    vol = nib.load(path).get_fdata().astype(np.float32)
    if vol.ndim == 4:
        vol = vol[..., 0]
    lo, hi = vol.min(), vol.max()
    if hi > lo:
        vol = (vol - lo) / (hi - lo)
    return vol


def patch_features(vol: np.ndarray, grid) -> np.ndarray:
    gx, gy, gz = grid
    Dx, Dy, Dz = vol.shape[:3]
    vol = vol[: (Dx // gx) * gx,
               : (Dy // gy) * gy,
               : (Dz // gz) * gz]
    px, py, pz = vol.shape[0]//gx, vol.shape[1]//gy, vol.shape[2]//gz

    feat = np.zeros((gx * gy * gz, N_FEATURES), dtype=np.float32)
    idx  = 0
    for ix in range(gx):
        for iy in range(gy):
            for iz in range(gz):
                p = vol[ix*px:(ix+1)*px,
                        iy*py:(iy+1)*py,
                        iz*pz:(iz+1)*pz].ravel()
                if p.std() >= 1e-9:
                    feat[idx, 0] = p.mean()
                    feat[idx, 1] = p.std()
                    feat[idx, 2] = p.min()
                    feat[idx, 3] = p.max()
                    feat[idx, 4] = np.percentile(p, 25)
                    feat[idx, 5] = np.percentile(p, 75)
                    feat[idx, 6] = float(scipy_stats.skew(p))
                    feat[idx, 7] = float(scipy_stats.kurtosis(p))
                idx += 1

    return np.nan_to_num(feat, nan=0.0, posinf=1.0, neginf=-1.0)


def _make_6conn_edges(grid):
    gx, gy, gz = grid
    def nid(x, y, z): return x * gy * gz + y * gz + z
    rows, cols = [], []
    for x in range(gx):
        for y in range(gy):
            for z in range(gz):
                s = nid(x, y, z)
                for dx, dy, dz in [(1,0,0),(-1,0,0),(0,1,0),
                                    (0,-1,0),(0,0,1),(0,0,-1)]:
                    nx2, ny2, nz2 = x+dx, y+dy, z+dz
                    if 0 <= nx2 < gx and 0 <= ny2 < gy and 0 <= nz2 < gz:
                        rows.append(s); cols.append(nid(nx2,ny2,nz2))
    ei = torch.tensor([rows, cols], dtype=torch.long)
    ew = torch.ones(len(rows), dtype=torch.float32)
    return ei, ew


_EDGES: dict = {}
for _sc, _cfg in SCALE_CONFIGS.items():
    _ei, _ew = _make_6conn_edges(_cfg["grid"])
    _EDGES[_sc] = (_ei, _ew)
    print(f"Scale {_sc}: {_cfg['n_nodes']} nodes | {_ei.shape[1]} edges")
print()


def volume_to_graph(path: str, scale: int, label: int,
                    augment: bool = False) -> Data:
    vol  = load_volume(path)
    feat = patch_features(vol, SCALE_CONFIGS[scale]["grid"])

    if augment:
        noise = np.random.randn(*feat.shape).astype(np.float32)
        feat  = feat + SIGMA_NOISE * noise
        feat  = np.nan_to_num(feat, nan=0.0, posinf=1.0, neginf=-1.0)

    ei, ew = _EDGES[scale]
    data   = Data(
        x          = torch.tensor(feat, dtype=torch.float32),
        edge_index = ei,
        edge_attr  = ew,
        y          = torch.tensor([label], dtype=torch.long),
    )
    data.num_nodes = SCALE_CONFIGS[scale]["n_nodes"]
    return data


def build_fold_graphs(dataset, train_idx, test_idx, augment_train: bool):
    train_g = {128: [], 224: []}
    test_g  = {128: [], 224: []}

    for idx in train_idx:
        d = dataset[idx]
        for sc in (128, 224):
            train_g[sc].append(volume_to_graph(d[f"path_{sc}"], sc,
                                               d["label"], augment=False))
            if augment_train:
                for _ in range(N_AUGMENTS):
                    train_g[sc].append(
                        volume_to_graph(d[f"path_{sc}"], sc,
                                        d["label"], augment=True))

    for idx in test_idx:
        d = dataset[idx]
        for sc in (128, 224):
            test_g[sc].append(volume_to_graph(d[f"path_{sc}"], sc,
                                              d["label"], augment=False))

    return train_g, test_g


# ═══════════════════════════════════════════════════════════════════
# 3.  SPECTRAL GCN
# ═══════════════════════════════════════════════════════════════════

class SpectralGCN(nn.Module):
    def __init__(self, in_ch: int, hidden: int = HIDDEN,
                 K: int = K_CHEB, drop: float = DROPOUT):
        super().__init__()
        self.conv1 = ChebConv(in_ch, hidden, K=K)
        self.conv2 = ChebConv(hidden, hidden, K=K)
        self.fc    = nn.Linear(hidden, 2)
        self.drop  = drop

    def forward(self, x, edge_index, edge_weight, batch):
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, p=self.drop, training=self.training)
        x = F.relu(self.conv2(x, edge_index, edge_weight))
        x = F.dropout(x, p=self.drop, training=self.training)
        x = global_mean_pool(x, batch)
        return F.log_softmax(self.fc(x), dim=1)


# ═══════════════════════════════════════════════════════════════════
# 4.  TRAINING & METRIC HELPERS
# ═══════════════════════════════════════════════════════════════════

def _train_epoch(model, loader, opt):
    model.train()
    for data in loader:
        data = data.to(DEVICE)
        opt.zero_grad()
        out  = model(data.x, data.edge_index, data.edge_attr, data.batch)
        F.nll_loss(out, data.y.squeeze()).backward()
        opt.step()


@torch.no_grad()
def _predict(model, loader):
    model.eval()
    preds, probs, trues = [], [], []
    for data in loader:
        data = data.to(DEVICE)
        out  = model(data.x, data.edge_index, data.edge_attr, data.batch)
        preds.extend(out.argmax(1).cpu().numpy())
        probs.extend(torch.exp(out)[:, 1].cpu().numpy())
        trues.extend(data.y.squeeze().cpu().numpy())
    return np.array(preds), np.array(probs), np.array(trues)


def metrics(y_true, y_pred, y_prob):
    cm   = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = 0.5
    return dict(
        accuracy    = accuracy_score(y_true, y_pred),
        precision   = precision_score(y_true, y_pred, zero_division=0),
        f1          = f1_score(y_true, y_pred, zero_division=0),
        roc_auc     = auc,
        mcc         = matthews_corrcoef(y_true, y_pred),
        recall      = recall_score(y_true, y_pred, zero_division=0),
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        fm          = fowlkes_mallows_score(y_true, y_pred),
        npv         = tn / (tn + fn) if (tn + fn) > 0 else 0.0,
    )


def train_gcn(train_data, test_data, in_ch: int,
              tag: str = "", verbose: bool = True):
    ldr_tr = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    ldr_te = DataLoader(test_data,  batch_size=len(test_data), shuffle=False)

    model = SpectralGCN(in_ch).to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=LR,
                              weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    for _ in range(EPOCHS):
        _train_epoch(model, ldr_tr, opt)
        sched.step()

    y_pred, y_prob, y_true = _predict(model, ldr_te)
    met = metrics(y_true, y_pred, y_prob)
    if verbose:
        print(f"    {tag:<14s}  acc={met['accuracy']:.4f}  "
              f"prec={met['precision']:.4f}  f1={met['f1']:.4f}  "
              f"auc={met['roc_auc']:.4f}  mcc={met['mcc']:.4f}")
    return met, model, y_prob, y_true


# ═══════════════════════════════════════════════════════════════════
# 5.  MULTI-SCALE DECISION VOTING GCN
# ═══════════════════════════════════════════════════════════════════

def run_dv_gcn(train_graphs: dict, test_graphs: dict) -> dict:
    all_probs, y_true = [], None
    for sc in (128, 224):
        print(f"    [DV-GCN] scale={sc} …")
        met, _, probs, trues = train_gcn(
            train_graphs[sc], test_graphs[sc],
            in_ch=N_FEATURES, tag=f"scale-{sc}", verbose=True)
        all_probs.append(probs)
        if y_true is None:
            y_true = trues

    avg_prob  = np.mean(all_probs, axis=0)
    avg_pred  = (avg_prob >= 0.5).astype(int)
    return metrics(y_true, avg_pred, avg_prob)


# ═══════════════════════════════════════════════════════════════════
# 6.  LOGISTIC REGRESSION BASELINE
# ═══════════════════════════════════════════════════════════════════

def _lr_features(dataset, indices, scales):
    blocks = []
    for sc in scales:
        rows = []
        for i in indices:
            d = dataset[i]
            vol  = load_volume(d[f"path_{sc}"])
            feat = patch_features(vol, SCALE_CONFIGS[sc]["grid"])
            rows.append(feat.ravel())
        blocks.append(np.array(rows, dtype=np.float32))
    return np.concatenate(blocks, axis=1)


def run_logistic(dataset, y, train_idx, test_idx, scales):
    X_tr = _lr_features(dataset, train_idx, scales)
    X_te = _lr_features(dataset, test_idx,  scales)
    y_tr, y_te = y[train_idx], y[test_idx]
    sc   = StandardScaler()
    X_tr = sc.fit_transform(X_tr)
    X_te = sc.transform(X_te)
    clf  = LogisticRegression(max_iter=3000, C=1.0, solver="lbfgs",
                              random_state=SEED)
    clf.fit(X_tr, y_tr)
    return metrics(y_te, clf.predict(X_te), clf.predict_proba(X_te)[:, 1])


# ═══════════════════════════════════════════════════════════════════
# 7.  RESULTS TABLE
# ═══════════════════════════════════════════════════════════════════

def aggregate(fold_results):
    return {k: (np.mean([m[k] for m in fold_results]),
                np.std( [m[k] for m in fold_results]))
            for k in fold_results[0]}


def print_results_table(title, rows):
    COLS = ["Accuracy", "Precision", "F1", "ROC AUC", "MCC",
            "Recall", "Specificity", "F-M Index", "NPV"]
    KEYS = ["accuracy", "precision", "f1", "roc_auc", "mcc",
            "recall", "specificity", "fm", "npv"]

    records = []
    for modality, scale_str, agg in rows:
        row = {"Modality": modality, "Scale": scale_str}
        for col, key in zip(COLS, KEYS):
            mu, sd = agg[key]
            row[col] = f"{mu:.4f} ± {sd:.4f}"
        records.append(row)

    df  = pd.DataFrame(records)
    bar = "═" * 210
    print(f"\n{bar}\n  {title}\n{bar}")
    print(df.to_string(index=False, col_space=20))
    return df


# ═══════════════════════════════════════════════════════════════════
# 8.  MAIN: 5-FOLD CROSS-VALIDATION
# ═══════════════════════════════════════════════════════════════════

def main():
    dataset, y = build_dataset(BASE_PATH, SITES)

    if len(dataset) < 10:
        raise RuntimeError(
            f"Too few subjects ({len(dataset)}). "
            "Check phenotypic TSV files and NIfTI paths."
        )

    skf    = StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                             random_state=SEED)
    splits = list(skf.split(np.zeros(len(dataset)), y))

    gcn_res = {128: [], 224: [], "multi": []}
    lr_res  = {128: [], 224: [], "multi": []}

    for fold, (tr_idx, te_idx) in enumerate(splits):
        print(f"\n{'━'*70}")
        print(f"  FOLD {fold+1}/{N_FOLDS}   "
              f"train={len(tr_idx)}  test={len(te_idx)}")
        print(f"{'━'*70}")

        print("  Building single-scale graphs (no aug) …")
        tr_single, te_single = build_fold_graphs(
            dataset, tr_idx, te_idx, augment_train=False)

        print(f"  Building multi-scale graphs (aug ×{N_AUGMENTS+1}) …")
        tr_multi, te_multi = build_fold_graphs(
            dataset, tr_idx, te_idx, augment_train=True)

        print("\n  Single-scale GCN baselines …")
        for sc in (128, 224):
            met, _, _, _ = train_gcn(
                tr_single[sc], te_single[sc],
                in_ch=N_FEATURES, tag=f"GCN scale-{sc}")
            gcn_res[sc].append(met)

        print("\n  Multi-scale DV-GCN (proposed) …")
        met = run_dv_gcn(tr_multi, te_multi)
        gcn_res["multi"].append(met)
        print(f"\n  → DV-GCN  acc={met['accuracy']:.4f}  "
              f"f1={met['f1']:.4f}  "
              f"auc={met['roc_auc']:.4f}  "
              f"mcc={met['mcc']:.4f}")

        print("\n  Logistic Regression baselines …")
        for sc in (128, 224):
            met = run_logistic(dataset, y, tr_idx, te_idx, [sc])
            lr_res[sc].append(met)
            print(f"    LR scale-{sc:<3d}  "
                  f"acc={met['accuracy']:.4f}  f1={met['f1']:.4f}")
        met = run_logistic(dataset, y, tr_idx, te_idx, [128, 224])
        lr_res["multi"].append(met)
        print(f"    LR multi     "
              f"acc={met['accuracy']:.4f}  f1={met['f1']:.4f}")

    df_gcn = print_results_table(
        "TABLE I — GCN MODELS  (ADHD-200 · scales 128 & 224)",
        [
            ("Single",        "128×128",
             aggregate(gcn_res[128])),
            ("Single",        "224×224",
             aggregate(gcn_res[224])),
            ("Multi (DV-GCN)","128×128 & 224×224",
             aggregate(gcn_res["multi"])),
        ],
    )

    df_lr = print_results_table(
        "TABLE II — LOGISTIC REGRESSION BASELINE  (ADHD-200 · scales 128 & 224)",
        [
            ("Single", "128×128",          aggregate(lr_res[128])),
            ("Single", "224×224",          aggregate(lr_res[224])),
            ("Multi",  "128×128 & 224×224",aggregate(lr_res["multi"])),
        ],
    )

    out = "/kaggle/working"
    df_gcn.to_csv(f"{out}/table1_gcn_adhd200.csv",      index=False)
    df_lr.to_csv( f"{out}/table2_logistic_adhd200.csv", index=False)
    print("\nSaved → table1_gcn_adhd200.csv | table2_logistic_adhd200.csv")


if __name__ == "__main__":
    main()
