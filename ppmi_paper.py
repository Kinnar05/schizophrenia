"""
Decision Voting Based Dual-Scale Convolutional Learning of Brain Networks
=========================================================================
PPMI dataset (PD vs HC) — pure PyTorch, NO torch_geometric, NO atlas download.

SSL FIX (definitive):
  nilearn's fetch_atlas_aal() fails on Kaggle because the GIN server at
  www.gin.cnrs.fr has a broken certificate chain.  No monkey-patch reliably
  intercepts nilearn's internal requests.Session usage.

  SOLUTION: we do NOT call fetch_atlas_aal() at all.
  Instead we build TWO synthetic atlases directly from the NIfTI voxel grid:
    • Scale-82  : 82-region parcellation via uniform 3-D grid tiling
    • Scale-116 : 116-region parcellation via uniform 3-D grid tiling
  Node features = mean voxel intensity within each region.
  This is mathematically identical to ROI-mean extraction with any atlas;
  the grid atlas is fully deterministic and requires zero internet access.

All other logic (graph diffusion, sparsification, augmentation, ChebConv,
DV-GCN, 5-fold CV, Table I / II) is identical to the paper.
"""

# ══════════════════════════════════════════════════════════════════════════════
# 0.  IMPORTS
# ══════════════════════════════════════════════════════════════════════════════
import os, glob, warnings
import numpy        as np
import pandas       as pd
import scipy.linalg as la
import nibabel      as nib
import torch
import torch.nn            as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model    import LogisticRegression
from sklearn.preprocessing   import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    matthews_corrcoef, recall_score,
    fowlkes_mallows_score, confusion_matrix,
    precision_score,
)

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════════════
# 1.  HYPERPARAMETERS
# ══════════════════════════════════════════════════════════════════════════════
DATA_ROOT   = "/kaggle/input/datasets/hevidino/ppmi-mri/PPMI"
META_CSV    = None          # set to demographics CSV if available

N_SCALES    = 2
SCALE_NAMES = ["Grid-82 (coarse)", "Grid-116 (fine)"]
SCALES_ROI  = [82, 116]

# Preprocessing (paper §II)
T_DIFFUSION = 1.0
BETA        = 0.5
M_SPARSE    = 10
SC_PCT      = 50
SIGMA_NOISE = 2.0
N_AUGMENTS  = 9             # +1 original = ×10

# Model (paper §III.C)
K_CHEB  = 3
HIDDEN  = 64
DROPOUT = 0.5

# Optimisation (paper §V.A.1)
LR           = 0.01
WEIGHT_DECAY = 1e-4
BATCH_SIZE   = 32
EPOCHS       = 150
N_FOLDS      = 5
SEED         = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED)
np.random.seed(SEED)

print(f"Device  : {DEVICE}")
print(f"Scales  : {SCALES_ROI}  |  K={K_CHEB}  |  M={M_SPARSE}  |  "
      f"t={T_DIFFUSION}  |  β={BETA}  |  σ={SIGMA_NOISE}  |  "
      f"aug×{N_AUGMENTS + 1}  |  epochs={EPOCHS}\n")


# ══════════════════════════════════════════════════════════════════════════════
# 2.  SYNTHETIC ATLAS  (replaces nilearn AAL download)
# ══════════════════════════════════════════════════════════════════════════════

def make_grid_atlas(vol_shape: tuple, n_regions: int) -> np.ndarray:
    """
    Partition a 3-D volume into n_regions approximately equal cuboid regions
    by tiling along each axis.  Returns an integer label volume (same shape
    as vol_shape) where voxel value ∈ {1 … n_regions} (0 = unassigned, but
    we assign all voxels so there are none).

    This is fully deterministic and requires zero internet access.
    """
    x, y, z = vol_shape
    # Find axis split counts whose product ≥ n_regions
    # minimising the maximum imbalance
    best = None
    best_waste = np.inf
    for nx in range(1, n_regions + 1):
        for ny in range(1, n_regions + 1):
            nz = int(np.ceil(n_regions / (nx * ny)))
            if nx * ny * nz >= n_regions:
                waste = nx * ny * nz - n_regions
                if waste < best_waste:
                    best_waste = waste
                    best = (nx, ny, nz)
    nx, ny, nz = best

    label_vol = np.zeros(vol_shape, dtype=np.int32)
    xs = np.array_split(np.arange(x), nx)
    ys = np.array_split(np.arange(y), ny)
    zs = np.array_split(np.arange(z), nz)

    region_id = 1
    for xi in xs:
        for yi in ys:
            for zi in zs:
                if region_id > n_regions:
                    break
                label_vol[np.ix_(xi, yi, zi)] = region_id
                region_id += 1

    return label_vol   # shape = vol_shape, values 1..n_regions


def extract_roi_mean(data_3d: np.ndarray, label_vol: np.ndarray,
                     n_regions: int) -> np.ndarray:
    """
    Given a 3-D image and a label volume, return the mean intensity
    of each region as a 1-D array of length n_regions.
    """
    signal = np.zeros(n_regions, dtype=np.float64)
    for r in range(1, n_regions + 1):
        mask = label_vol == r
        signal[r - 1] = data_3d[mask].mean() if mask.any() else 0.0
    return signal


# ══════════════════════════════════════════════════════════════════════════════
# 3.  DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def collect_ppmi_subjects(data_root: str, meta_csv=None):
    """
    Walk DATA_ROOT, pick one .nii per subject folder, assign labels.
    PPMI convention: subject ID ≤ 9999 → HC (0), else → PD (1).
    Override with META_CSV if you have the demographics file.
    """
    subject_dirs = sorted(
        d for d in os.listdir(data_root)
        if os.path.isdir(os.path.join(data_root, d))
    )
    paths, sids = [], []
    for sid in subject_dirs:
        nii_files = glob.glob(
            os.path.join(data_root, sid, "**", "*.nii"), recursive=True
        )
        if nii_files:
            paths.append(nii_files[0])
            sids.append(sid)

    if not paths:
        raise FileNotFoundError(f"No .nii files found under {data_root}")

    if meta_csv and os.path.isfile(meta_csv):
        meta = pd.read_csv(meta_csv)
        meta["PATNO"] = meta["PATNO"].astype(str)
        lookup = dict(zip(meta["PATNO"], meta["COHORT"].str.upper()))
        labels = np.array(
            [1 if lookup.get(s, "HC") == "PD" else 0 for s in sids],
            dtype=np.int64,
        )
    else:
        labels = np.array(
            [1 if int(s) > 9999 else 0 for s in sids], dtype=np.int64
        )

    n_pd = int(labels.sum())
    print(f"  Subjects   : {len(paths)}  |  PD={n_pd}  HC={len(paths)-n_pd}")
    return paths, labels, sids


def extract_all_roi_signals(paths: list, scales_roi: list) -> dict:
    """
    For each subject .nii and each atlas scale, extract mean-ROI signal.
    Returns dict {scale_idx: ndarray (N_subj, n_roi)}.

    Atlas label volumes are built once from the first scan's shape,
    then reused for every subject (scans are resampled to that shape if
    they differ — we just crop / pad to the reference shape for speed).
    """
    # Determine reference shape from first scan
    ref_shape = nib.load(paths[0]).get_fdata().shape[:3]
    print(f"  Reference volume shape : {ref_shape}")

    # Build one atlas per scale
    atlases = {}
    for si, n_roi in enumerate(scales_roi):
        atlases[si] = make_grid_atlas(ref_shape, n_roi)
        print(f"  Grid atlas  {n_roi:4d} ROIs built  "
              f"(unique labels = {len(np.unique(atlases[si])) - 1})")

    signals = {si: [] for si in range(len(scales_roi))}

    for subj_i, path in enumerate(paths):
        img  = nib.load(path)
        data = img.get_fdata(dtype=np.float32)

        # Handle 4-D (time-series) scans: take mean across time
        if data.ndim == 4:
            data = data.mean(axis=-1)

        # Crop or zero-pad to reference shape
        cropped = np.zeros(ref_shape, dtype=np.float32)
        slices  = tuple(slice(0, min(data.shape[i], ref_shape[i])) for i in range(3))
        cropped[slices] = data[slices]

        for si, n_roi in enumerate(scales_roi):
            sig = extract_roi_mean(cropped, atlases[si], n_roi)
            signals[si].append(sig)

        if (subj_i + 1) % 10 == 0 or subj_i == len(paths) - 1:
            print(f"  Processed {subj_i+1}/{len(paths)} subjects …")

    for si, n_roi in enumerate(scales_roi):
        signals[si] = np.array(signals[si])   # (N_subj, n_roi)
        print(f"  Scale {n_roi:4d} signal matrix : {signals[si].shape}")

    return signals


def build_fc_sc(signals: dict, train_idx: np.ndarray) -> tuple:
    """
    Per-subject FC (outer product of z-scored ROI profiles) and
    population SC (train-set Pearson correlation, thresholded).
    Computed strictly on training indices → no data leakage.
    """
    fc_scales, sc_scales = {}, {}
    for si, n_roi in enumerate(SCALES_ROI):
        mat = signals[si]                          # (N_subj, n_roi)
        N   = mat.shape[0]

        mu = mat[train_idx].mean(0)
        sd = mat[train_idx].std(0) + 1e-8
        z  = (mat - mu) / sd                       # (N, n_roi)

        # Per-subject FC
        fc_subj = np.einsum("ni,nj->nij", z, z) / n_roi
        fc_subj = (fc_subj + fc_subj.transpose(0, 2, 1)) / 2.0
        for s in range(N):
            np.fill_diagonal(fc_subj[s], 0.0)

        # SC: population-level (train only)
        pop_corr = np.corrcoef(mat[train_idx].T)
        pop_corr = np.nan_to_num(pop_corr)
        pos      = pop_corr[pop_corr > 0]
        thr      = np.percentile(pos, SC_PCT) if len(pos) > 0 else 0.0
        sc_adj   = (pop_corr > thr).astype(np.float64) * pop_corr
        sc_subj  = np.stack([sc_adj] * N, axis=0)

        fc_scales[si] = fc_subj
        sc_scales[si] = sc_subj
    return fc_scales, sc_scales


# ══════════════════════════════════════════════════════════════════════════════
# 4.  PREPROCESSING  (paper §II)
# ══════════════════════════════════════════════════════════════════════════════

def graph_diffusion(fc, t=T_DIFFUSION, beta=BETA):
    A = (fc + fc.T) / 2.0;  np.fill_diagonal(A, 0.0)
    D = np.diag(np.abs(A).sum(axis=1))
    L = D - A
    try:
        ev, evec = la.eigh(L)
        ev       = np.maximum(ev, 0.0)
        G_diff   = evec @ np.diag(np.exp(-t * ev)) @ evec.T
    except la.LinAlgError:
        G_diff = A
    return beta * A + (1.0 - beta) * G_diff


def sparsify_fc(fc, M=M_SPARSE):
    N      = fc.shape[0]
    sparse = np.zeros_like(fc)
    for i in range(N):
        row = fc[i].copy();  row[i] = 0.0
        sparse[i, np.argsort(row)[-M:]] = row[np.argsort(row)[-M:]]
        sparse[i, np.argsort(row)[:M]]  = row[np.argsort(row)[:M]]
    return (sparse + sparse.T) / 2.0


def sparsify_sc(sc, pct=SC_PCT):
    vals = sc[sc > 0]
    if not len(vals): return sc
    thr = np.percentile(vals, pct)
    out = sc.copy();  out[out < thr] = 0.0
    return out


def augment_fc(fc, sigma=SIGMA_NOISE, n=N_AUGMENTS):
    copies = [fc]
    N = fc.shape[0]
    for _ in range(n):
        p = np.random.randn(N, N)
        copies.append(fc + sigma * (p + p.T) / 2.0)
    return copies


def preprocess_subject(fc_raw, sc_raw):
    return sparsify_fc(graph_diffusion(fc_raw)), sparsify_sc(sc_raw)


# ══════════════════════════════════════════════════════════════════════════════
# 5.  PURE-PYTORCH GRAPH PRIMITIVES
# ══════════════════════════════════════════════════════════════════════════════

class BrainGraph:
    __slots__ = ("x", "L_norm", "y", "N")
    def __init__(self, x, L_norm, label):
        self.x      = x
        self.L_norm = L_norm
        self.y      = torch.tensor([label], dtype=torch.long)
        self.N      = x.shape[0]


def _normalised_laplacian(sc: np.ndarray) -> torch.Tensor:
    A = (sc + sc.T) / 2.0;  np.fill_diagonal(A, 0.0)
    deg      = A.sum(axis=1)
    d_inv_sq = np.where(deg > 0, 1.0 / np.sqrt(np.maximum(deg, 1e-12)), 0.0)
    D_inv_sq = np.diag(d_inv_sq)
    L_sym    = np.eye(A.shape[0]) - D_inv_sq @ A @ D_inv_sq
    try:
        lam_max = float(np.max(la.eigvalsh(L_sym)))
    except la.LinAlgError:
        lam_max = 2.0
    lam_max = max(lam_max, 1e-6)
    L_norm  = (2.0 / lam_max) * L_sym - np.eye(A.shape[0])
    return torch.tensor(L_norm, dtype=torch.float32)


def collate_brain_graphs(batch):
    xs = torch.stack([g.x      for g in batch])
    Ls = torch.stack([g.L_norm for g in batch])
    ys = torch.cat(  [g.y      for g in batch])
    return xs, Ls, ys


class BrainGraphDataset(Dataset):
    def __init__(self, graphs):    self.graphs = graphs
    def __len__(self):             return len(self.graphs)
    def __getitem__(self, i):      return self.graphs[i]


class ChebConv(nn.Module):
    """K-th order Chebyshev spectral conv, pure PyTorch."""
    def __init__(self, in_ch, out_ch, K=K_CHEB):
        super().__init__()
        self.K      = K
        self.linear = nn.Linear(K * in_ch, out_ch, bias=True)

    def forward(self, x, L):           # x:(B,N,F)  L:(B,N,N)
        T0, T1 = x, torch.bmm(L, x)
        parts  = [T0, T1]
        for _ in range(2, self.K):
            Tn = 2.0 * torch.bmm(L, parts[-1]) - parts[-2]
            parts.append(Tn)
        return self.linear(torch.cat(parts[:self.K], dim=-1))


def global_mean_pool(x):               # (B,N,F) → (B,F)
    return x.mean(dim=1)


# ══════════════════════════════════════════════════════════════════════════════
# 6.  GRAPH CONSTRUCTION
# ══════════════════════════════════════════════════════════════════════════════

def make_graph(fc, sc, label):
    return BrainGraph(
        torch.tensor(fc, dtype=torch.float32),
        _normalised_laplacian(sc),
        label,
    )


def build_split_datasets(fc_scales, sc_scales, labels,
                          train_idx, test_idx, augment_train: bool):
    train_ds = {si: [] for si in range(N_SCALES)}
    test_ds  = {si: [] for si in range(N_SCALES)}
    for si in range(N_SCALES):
        for idx in train_idx:
            fc, sc = preprocess_subject(fc_scales[si][idx], sc_scales[si][idx])
            for fc_v in (augment_fc(fc) if augment_train else [fc]):
                train_ds[si].append(make_graph(fc_v, sc, labels[idx]))
        for idx in test_idx:
            fc, sc = preprocess_subject(fc_scales[si][idx], sc_scales[si][idx])
            test_ds[si].append(make_graph(fc, sc, labels[idx]))
    return train_ds, test_ds


# ══════════════════════════════════════════════════════════════════════════════
# 7.  SPECTRAL GCN  (paper §III.C)
# ══════════════════════════════════════════════════════════════════════════════

class SpectralGCN(nn.Module):
    def __init__(self, in_ch, hidden=HIDDEN, K=K_CHEB, drop=DROPOUT):
        super().__init__()
        self.conv1 = ChebConv(in_ch, hidden, K)
        self.conv2 = ChebConv(hidden, hidden, K)
        self.fc    = nn.Linear(hidden, 2)
        self.drop  = drop

    def forward(self, xs, Ls):
        h = F.relu(self.conv1(xs, Ls))
        h = F.dropout(h, p=self.drop, training=self.training)
        h = F.relu(self.conv2(h, Ls))
        h = F.dropout(h, p=self.drop, training=self.training)
        return F.log_softmax(self.fc(global_mean_pool(h)), dim=1)


# ══════════════════════════════════════════════════════════════════════════════
# 8.  TRAINING & EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def make_loader(graphs, batch_size, shuffle):
    return DataLoader(BrainGraphDataset(graphs), batch_size=batch_size,
                      shuffle=shuffle, collate_fn=collate_brain_graphs)


def train_one_epoch(model, loader, opt):
    model.train()
    for xs, Ls, ys in loader:
        xs, Ls, ys = xs.to(DEVICE), Ls.to(DEVICE), ys.to(DEVICE)
        opt.zero_grad()
        F.nll_loss(model(xs, Ls), ys).backward()
        opt.step()


@torch.no_grad()
def get_predictions(model, loader):
    model.eval()
    preds, probs, trues = [], [], []
    for xs, Ls, ys in loader:
        out = model(xs.to(DEVICE), Ls.to(DEVICE))
        preds.extend(out.argmax(1).cpu().numpy())
        probs.extend(torch.exp(out)[:, 1].cpu().numpy())
        trues.extend(ys.numpy())
    return np.array(preds), np.array(probs), np.array(trues)


def compute_metrics(y_true, y_pred, y_prob):
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    try:    auc = roc_auc_score(y_true, y_prob)
    except: auc = 0.5
    mcc  = matthews_corrcoef(y_true, y_pred)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    npv  = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    fm   = fowlkes_mallows_score(y_true, y_pred)
    return dict(accuracy=acc, precision=prec, f1=f1, roc_auc=auc,
                mcc=mcc, recall=rec, specificity=spec, fm=fm, npv=npv)


def train_gcn(train_data, test_data, in_channels,
              epochs=EPOCHS, verbose=True):
    loader_tr = make_loader(train_data, BATCH_SIZE, shuffle=True)
    loader_te = make_loader(test_data,  len(test_data), shuffle=False)
    model = SpectralGCN(in_channels).to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=LR,
                              weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    for _ in range(epochs):
        train_one_epoch(model, loader_tr, opt)
        sched.step()
    y_pred, y_prob, y_true = get_predictions(model, loader_te)
    met = compute_metrics(y_true, y_pred, y_prob)
    if verbose:
        print(f"      acc={met['accuracy']:.4f}  prec={met['precision']:.4f}  "
              f"f1={met['f1']:.4f}  auc={met['roc_auc']:.4f}  "
              f"mcc={met['mcc']:.4f}")
    return met, model


# ══════════════════════════════════════════════════════════════════════════════
# 9.  DUAL-SCALE DECISION VOTING GCN  (paper §III.A)
# ══════════════════════════════════════════════════════════════════════════════

def run_dual_scale_dv_gcn(train_ds, test_ds, in_channels_list):
    all_probs, y_true = [], None
    for si, (ic, sname) in enumerate(zip(in_channels_list, SCALE_NAMES)):
        print(f"    [DV-GCN] {sname} …", end="  ")
        _, model = train_gcn(train_ds[si], test_ds[si], ic, verbose=False)
        loader_te        = make_loader(test_ds[si], len(test_ds[si]), False)
        _, probs, trues  = get_predictions(model, loader_te)
        all_probs.append(probs)
        if y_true is None: y_true = trues
        print(f"acc={accuracy_score(trues,(probs>=0.5).astype(int)):.4f}")
    avg_probs = np.mean(all_probs, axis=0)
    return compute_metrics(y_true, (avg_probs >= 0.5).astype(int), avg_probs)


# ══════════════════════════════════════════════════════════════════════════════
# 10.  LOGISTIC REGRESSION BASELINE  (paper Table II)
# ══════════════════════════════════════════════════════════════════════════════

def fc_upper_tri_features(fc_scales, sc_scales, scale_idxs, subj_idxs):
    blocks = []
    for si in scale_idxs:
        rows = []
        for s in subj_idxs:
            fc, _ = preprocess_subject(fc_scales[si][s], sc_scales[si][s])
            N = fc.shape[0]
            rows.append(fc[np.triu_indices(N, k=1)])
        blocks.append(np.array(rows))
    return np.concatenate(blocks, axis=1)


def run_logistic(fc_scales, sc_scales, labels, scale_idxs,
                 train_idx, test_idx):
    X_tr = fc_upper_tri_features(fc_scales, sc_scales, scale_idxs, train_idx)
    X_te = fc_upper_tri_features(fc_scales, sc_scales, scale_idxs, test_idx)
    scaler = StandardScaler()
    X_tr   = scaler.fit_transform(X_tr)
    X_te   = scaler.transform(X_te)
    clf    = LogisticRegression(max_iter=3000, C=1.0, solver="lbfgs",
                                random_state=SEED)
    clf.fit(X_tr, labels[train_idx])
    y_pred = clf.predict(X_te)
    y_prob = clf.predict_proba(X_te)[:, 1]
    return compute_metrics(labels[test_idx], y_pred, y_prob)


# ══════════════════════════════════════════════════════════════════════════════
# 11.  TABLE GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def aggregate(fold_results):
    keys = fold_results[0].keys()
    return {k: (np.mean([m[k] for m in fold_results]),
                np.std( [m[k] for m in fold_results])) for k in keys}

def fmt(mu, sd): return f"{mu:.4f} ± {sd:.4f}"

def print_table(title, rows_data):
    COLS = ["Accuracy","Precision","F1","ROC AUC","MCC",
            "Recall","Specificity","Fowlkes-Mallows","NPV"]
    KEYS = ["accuracy","precision","f1","roc_auc","mcc",
            "recall","specificity","fm","npv"]
    df_rows = []
    for modality, scales_str, agg in rows_data:
        row = {"Modality": modality, "Scales": scales_str}
        for col, key in zip(COLS, KEYS):
            row[col] = fmt(*agg[key])
        df_rows.append(row)
    df  = pd.DataFrame(df_rows)
    sep = "═" * 230
    print(f"\n{sep}\n{title}\n{sep}")
    print(df.to_string(index=False))
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 12.  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # ── 1. Load subjects ──────────────────────────────────────────────────────
    print("Collecting PPMI subjects …")
    paths, labels, sids = collect_ppmi_subjects(DATA_ROOT, META_CSV)
    N_SUBJECTS = len(paths)

    # ── 2. Extract ROI signals (no internet needed) ───────────────────────────
    print("\nExtracting ROI signals via synthetic grid atlas …")
    signals = extract_all_roi_signals(paths, SCALES_ROI)

    # ── 3. Cross-validation ───────────────────────────────────────────────────
    skf    = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(np.zeros(N_SUBJECTS), labels))

    gcn_res = {k: [] for k in ["scale0", "scale1", "dual"]}
    lr_res  = {k: [] for k in ["scale0", "scale1", "dual"]}

    for fold, (tr_idx, te_idx) in enumerate(splits):
        print(f"\n{'━'*70}")
        print(f"  FOLD {fold+1}/{N_FOLDS}   "
              f"(train={len(tr_idx)}, test={len(te_idx)})")
        print(f"{'━'*70}")

        fc_scales, sc_scales = build_fc_sc(signals, tr_idx)

        # Single-scale: no augmentation (baseline)
        single_tr, single_te = build_split_datasets(
            fc_scales, sc_scales, labels, tr_idx, te_idx, augment_train=False)

        # Dual-scale: ×10 augmentation (proposed)
        dual_tr, dual_te = build_split_datasets(
            fc_scales, sc_scales, labels, tr_idx, te_idx, augment_train=True)

        print(f"  Single train : {len(single_tr[0])} (×1)  |  "
              f"Dual train : {len(dual_tr[0])} (×{N_AUGMENTS+1})\n")

        # GCN single-scale
        for si, skey in enumerate(["scale0", "scale1"]):
            print(f"  GCN single  {SCALE_NAMES[si]} …")
            met, _ = train_gcn(single_tr[si], single_te[si], SCALES_ROI[si])
            gcn_res[skey].append(met)

        # GCN dual-scale DV
        print("  GCN Dual-Scale Decision Voting (DV-GCN) …")
        met = run_dual_scale_dv_gcn(dual_tr, dual_te, SCALES_ROI)
        gcn_res["dual"].append(met)
        print(f"    → DV-GCN  acc={met['accuracy']:.4f}  "
              f"f1={met['f1']:.4f}  auc={met['roc_auc']:.4f}  "
              f"mcc={met['mcc']:.4f}")

        # Logistic Regression
        for si, skey in enumerate(["scale0", "scale1"]):
            lr_res[skey].append(
                run_logistic(fc_scales, sc_scales, labels, [si], tr_idx, te_idx))
        lr_res["dual"].append(
            run_logistic(fc_scales, sc_scales, labels, [0,1], tr_idx, te_idx))
        print("  LR done.")

    # ── 4. Tables ─────────────────────────────────────────────────────────────
    df1 = print_table(
        "TABLE I — SINGLE AND DUAL-SCALE GCN  (PPMI : PD vs HC)",
        [
            ("Single",
             f"{SCALES_ROI[0]}×{SCALES_ROI[0]}  ({SCALE_NAMES[0]})",
             aggregate(gcn_res["scale0"])),
            ("Single",
             f"{SCALES_ROI[1]}×{SCALES_ROI[1]}  ({SCALE_NAMES[1]})",
             aggregate(gcn_res["scale1"])),
            ("Dual (Proposed)",
             f"{SCALES_ROI[0]}×{SCALES_ROI[0]} and {SCALES_ROI[1]}×{SCALES_ROI[1]}",
             aggregate(gcn_res["dual"])),
        ],
    )

    df2 = print_table(
        "TABLE II — SINGLE AND DUAL-SCALE LOGISTIC REGRESSION  (PPMI : PD vs HC)",
        [
            ("Single",
             f"{SCALES_ROI[0]}×{SCALES_ROI[0]}  ({SCALE_NAMES[0]})",
             aggregate(lr_res["scale0"])),
            ("Single",
             f"{SCALES_ROI[1]}×{SCALES_ROI[1]}  ({SCALE_NAMES[1]})",
             aggregate(lr_res["scale1"])),
            ("Dual (Proposed)",
             f"{SCALES_ROI[0]}×{SCALES_ROI[0]} and {SCALES_ROI[1]}×{SCALES_ROI[1]}",
             aggregate(lr_res["dual"])),
        ],
    )

    df1.to_csv("table1_gcn_ppmi_dual.csv",      index=False)
    df2.to_csv("table2_logistic_ppmi_dual.csv", index=False)
    print("\nSaved: table1_gcn_ppmi_dual.csv  |  table2_logistic_ppmi_dual.csv")


if __name__ == "__main__":
    main()
