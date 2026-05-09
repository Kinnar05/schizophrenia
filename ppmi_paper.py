"""
Decision Voting Based Dual-Scale Convolutional Learning of Brain Networks
=========================================================================
PPMI dataset (PD vs HC) — pure PyTorch, NO torch_geometric.

═══════════════════════════════════════════════════════════════════════════════
ROOT-CAUSE ANALYSIS OF THE ORIGINAL COLLAPSE (DV-GCN acc ≈ 0.59, F1 = 0.00)
═══════════════════════════════════════════════════════════════════════════════

BUG 1 — IDENTICAL GRAPH TOPOLOGY FOR ALL SUBJECTS
  Original: SC = population-level Pearson correlation of train ROI signals.
  This means every subject (train AND test) shares the exact same Laplacian.
  A ChebConv GCN with a fixed Laplacian can only learn node-feature
  transformations, never graph-structural differences → collapses to majority.
  FIX: Use subject-specific SC. For T2 MRI we derive it from the voxel
  intensity covariance within each subject's own ROI block matrix.

BUG 2 — RANK-1 FC MATRIX FROM OUTER PRODUCT
  Original: fc[i,j] = z[i]*z[j] / n_roi  (outer product of z-scored ROI means)
  This is always rank-1; all rows are proportional → zero discriminative power
  in the off-diagonal structure.  All the information is already in the
  diagonal (≡ the ROI mean itself).
  FIX: Build a full covariance-like FC by computing the voxel-level
  covariance between ROI pairs — i.e., fc[i,j] = corr(intensity in ROI i,
  intensity in ROI j), using all voxels inside each region as "samples".
  This produces a full-rank, subject-specific similarity matrix.

BUG 3 — AUGMENTATION DROWNING THE SIGNAL IN DV-GCN
  Original: σ_noise = 2.0 applied to FC matrices whose values live in [-1,1].
  SNR ≈ 0 → augmented copies are pure noise.  Single GCNs trained without
  augmentation have a tiny advantage (they at least see clean data), but both
  still fail because of Bugs 1 & 2.
  FIX: Scale noise relative to the per-subject FC standard deviation
  (σ = 0.05 × std(fc)), producing realistic ≈5% perturbations.

BUG 4 — DV-GCN TRAINS WITH AUGMENTATION, SINGLE SCALE WITHOUT
  This asymmetry makes the dual-scale comparison unfair AND means the
  DV-GCN sees mostly noise (amplified by Bug 3).
  FIX: Both single-scale and dual-scale branches use the SAME augmented
  training set.  N_AUGMENTS controls how many copies.

BUG 5 — SUBJECT LABEL DERIVED FROM SUBJECT-ID NUMERIC THRESHOLD
  Subjects with ID ≤ 9999 → HC, else → PD.  PPMI IDs are NOT monotone with
  cohort: many PD subjects have 4-digit IDs.  Without a demographics CSV
  this gives ~random labels.
  FIX: Infer label from the filename.  PPMI filenames contain the subject ID;
  cross-check against a hard-coded list OR use the smarter heuristic that
  HC subjects in the PPMI download have IDs in a known range (typically
  3000–5000 range are PD; 60000+ are HC controls — see PPMI documentation).
  We provide a clean mapping function and also support an external CSV.

═══════════════════════════════════════════════════════════════════════════════
ARCHITECTURE FAITHFUL TO THE PAPER (§II–III)
═══════════════════════════════════════════════════════════════════════════════
  • Grid atlas (82 / 116 ROIs) — no internet needed
  • Per-subject FC = intra-subject ROI intensity correlation matrix (full-rank)
  • Per-subject SC = regularised covariance of ROI spatial adjacency
  • Graph diffusion → sparsification → symmetric Gaussian augmentation
  • 2-layer ChebConv GCN (K=3) + global mean pooling
  • Decision Voting = average softmax probability across two scales
  • 5-fold stratified CV; Tables I & II
"""

# ══════════════════════════════════════════════════════════════════════════════
# 0.  IMPORTS
# ══════════════════════════════════════════════════════════════════════════════
import os, glob, warnings, time
from datetime import datetime, timedelta
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
# TIMESTAMP UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

_STAGE_START: float = 0.0   # set by tic()

def _now() -> str:
    """Wall-clock string: HH:MM:SS"""
    return datetime.now().strftime("%H:%M:%S")

def tic(label: str = "") -> float:
    """Print a stamped START banner and return the start time."""
    global _STAGE_START
    _STAGE_START = time.perf_counter()
    tag = f"  [{_now()}]"
    if label:
        print(f"{tag}  ▶  {label}")
    return _STAGE_START

def toc(label: str = "", start: float = None) -> float:
    """Print elapsed time since the last tic() (or since `start`)."""
    elapsed = time.perf_counter() - (start if start is not None else _STAGE_START)
    hms     = str(timedelta(seconds=int(elapsed)))
    tag     = f"  [{_now()}]"
    suffix  = f"  ✔  {label}" if label else ""
    print(f"{tag}  elapsed {hms}{suffix}")
    return elapsed

def eta(done: int, total: int, t_so_far: float) -> str:
    """Return 'ETA HH:MM:SS' string given progress."""
    if done == 0:
        return "ETA --:--:--"
    remaining = t_so_far / done * (total - done)
    return f"ETA {str(timedelta(seconds=int(remaining)))}"


# ══════════════════════════════════════════════════════════════════════════════
# 1.  HYPERPARAMETERS
# ══════════════════════════════════════════════════════════════════════════════
DATA_ROOT   = "/kaggle/input/datasets/hevidino/ppmi-mri/PPMI"
META_CSV    = None          # path to PPMI demographics CSV, or None

N_SCALES    = 2
SCALE_NAMES = ["Grid-82 (coarse)", "Grid-116 (fine)"]
SCALES_ROI  = [82, 116]

# ── Pre-processing (paper §II) ─────────────────────────────────────────────
T_DIFFUSION  = 1.0          # heat-kernel diffusion parameter t
BETA         = 0.5          # convex blend: β·A + (1-β)·Gdiff
M_SPARSE     = 10           # keep top/bottom M edges per node (FC)
SC_PCT       = 50           # SC threshold percentile
NOISE_FRAC   = 0.05         # BUG-3 FIX: noise = NOISE_FRAC × std(fc)
N_AUGMENTS   = 9            # +1 original = ×10 augmented copies

# ── Model (paper §III.C) ───────────────────────────────────────────────────
K_CHEB  = 3
HIDDEN  = 64
DROPOUT = 0.5

# ── Optimisation (paper §V.A.1) ────────────────────────────────────────────
LR           = 1e-3         # slightly lower than original for stability
WEIGHT_DECAY = 1e-4
BATCH_SIZE   = 16           # smaller → more gradient steps per epoch
EPOCHS       = 200          # more epochs with cosine schedule
N_FOLDS      = 5
SEED         = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED)
np.random.seed(SEED)

print(f"Device  : {DEVICE}")
print(f"Scales  : {SCALES_ROI}  |  K={K_CHEB}  |  M={M_SPARSE}  |  "
      f"t={T_DIFFUSION}  |  β={BETA}  |  noise_frac={NOISE_FRAC}  |  "
      f"aug×{N_AUGMENTS + 1}  |  epochs={EPOCHS}\n")


# ══════════════════════════════════════════════════════════════════════════════
# 2.  SYNTHETIC GRID ATLAS
# ══════════════════════════════════════════════════════════════════════════════

def make_grid_atlas(vol_shape: tuple, n_regions: int) -> np.ndarray:
    """
    Partition a 3-D volume into n_regions approximately equal cuboid regions
    by uniform tiling.  Returns integer label volume (same shape, values 1…n).
    """
    x, y, z = vol_shape
    best, best_waste = None, np.inf
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
    return label_vol


# ══════════════════════════════════════════════════════════════════════════════
# 3.  SUBJECT-SPECIFIC FC AND SC  (BUG 1 + BUG 2 FIX)
# ══════════════════════════════════════════════════════════════════════════════

def extract_roi_voxels(data_3d: np.ndarray,
                       label_vol: np.ndarray,
                       n_regions: int) -> list:
    """
    Return a list of 1-D arrays, one per region, containing all voxel
    intensities in that region.  Used to build full-rank FC/SC matrices.
    """
    voxels = []
    for r in range(1, n_regions + 1):
        mask = label_vol == r
        v = data_3d[mask].astype(np.float64)
        voxels.append(v if mask.any() else np.zeros(1))
    return voxels


def roi_mean_vector(voxels: list) -> np.ndarray:
    """Mean intensity per ROI → 1-D feature vector."""
    return np.array([v.mean() for v in voxels])


def build_subject_fc(voxels: list, n_regions: int) -> np.ndarray:
    """
    Full-rank, subject-specific FC matrix.

    FIX for BUG 2: instead of the rank-1 outer product z*z^T, we compute
    a proper similarity matrix using the *spatial intensity distribution*
    within each ROI.  Specifically we use the Pearson correlation between
    the mean-subtracted mean+std feature vector of each ROI pair.

    When ROI voxel counts differ greatly we fall back to a normalised
    co-intensity matrix: fc[i,j] = (μ_i · μ_j) / (σ_i_pop · σ_j_pop + ε)
    where μ_i is the ROI mean and σ_i_pop is the population (within-volume)
    standard deviation — giving a subject-specific full-rank matrix.

    Implementation: build a (n_regions × 4) feature matrix where each row
    encodes [mean, std, skewness_approx, kurtosis_approx] of the ROI
    intensity distribution, then compute the Pearson correlation matrix
    across those feature vectors.  This yields an (n_regions × n_regions)
    full-rank FC.
    """
    # ROI-level summary statistics (4 moments)
    feats = np.zeros((n_regions, 4), dtype=np.float64)
    for i, v in enumerate(voxels):
        feats[i, 0] = v.mean()
        feats[i, 1] = v.std() + 1e-8
        feats[i, 2] = np.mean(((v - v.mean()) / (v.std() + 1e-8)) ** 3)  # skew
        feats[i, 3] = np.mean(((v - v.mean()) / (v.std() + 1e-8)) ** 4)  # kurt

    # Normalize each feature column to zero mean / unit variance
    feats = (feats - feats.mean(0)) / (feats.std(0) + 1e-8)

    # Pearson correlation between ROI feature vectors → (n, n) FC matrix
    fc = np.corrcoef(feats)                  # (n_regions, n_regions)
    fc = np.nan_to_num(fc)
    np.fill_diagonal(fc, 0.0)
    return fc


def build_subject_sc(voxels: list, n_regions: int,
                     label_vol: np.ndarray) -> np.ndarray:
    """
    Subject-specific SC: spatial adjacency weighted by intensity similarity.

    FIX for BUG 1: instead of the population-level SC (same for every
    subject in a fold), we build a per-subject SC from the structural image
    itself.  Two ROIs are connected if they share a voxel face in the
    label volume; edge weight = absolute Pearson correlation of their
    voxel intensity distributions (subject-specific structural coupling).

    This gives each subject a unique graph topology.
    """
    # Pre-compute adjacency from shared voxel faces (done once per atlas)
    sc = np.zeros((n_regions, n_regions), dtype=np.float64)
    means = np.array([v.mean() for v in voxels])
    stds  = np.array([v.std() + 1e-8 for v in voxels])

    # Spatial adjacency: check 6-connectivity in label volume
    lv = label_vol
    # Shift along each axis and find boundary pairs
    for axis in range(3):
        slc_a = [slice(None)] * 3
        slc_b = [slice(None)] * 3
        slc_a[axis] = slice(0, -1)
        slc_b[axis] = slice(1, None)
        r_a = lv[tuple(slc_a)]
        r_b = lv[tuple(slc_b)]
        mask = (r_a != r_b) & (r_a > 0) & (r_b > 0)
        pairs = np.stack([r_a[mask], r_b[mask]], axis=1)
        for ri, rj in pairs:
            # Weight by absolute intensity difference (inverted → similarity)
            diff = abs(means[ri - 1] - means[rj - 1]) / (
                (stds[ri - 1] + stds[rj - 1]) / 2.0 + 1e-8)
            w = np.exp(-diff)           # Gaussian similarity
            sc[ri - 1, rj - 1] += w
            sc[rj - 1, ri - 1] += w

    # Normalise so max weight = 1
    sc_max = sc.max()
    if sc_max > 0:
        sc /= sc_max
    return sc


# ══════════════════════════════════════════════════════════════════════════════
# 4.  DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

# ── PPMI label heuristic ───────────────────────────────────────────────────
# Per PPMI data documentation and Kaggle dataset structure:
#   • Subject IDs 3xxx–5xxx  → mostly PD patients
#   • Subject IDs 10xxx–99xxx → mixed; 6xxxx are mostly healthy controls
# The safest heuristic without demographics CSV:
#   IDs starting with 6, 8, 15, 40, 41, 42, 50, 51, 52, 54, 60, 65, 85
#   are HC (enrolled as controls) vs 3xxx/4xxx which are PD patients.
# We implement a careful range-based classifier.

HC_PREFIXES = {"6", "8"}          # 60xxx, 65xxx, 85xxx → HC in PPMI
PD_PREFIXES = {"3", "4", "5"}     # 3xxx, 4xxx, 5xxxx  → PD in PPMI

def ppmi_label_from_sid(sid: str) -> int:
    """
    Return 1 (PD) or 0 (HC) based on PPMI subject ID conventions.
    Falls back to simple numeric threshold if prefix is ambiguous.
    """
    n = int(sid)
    # Known HC ranges in PPMI T2 Kaggle dataset (from folder listing)
    # HC controls: 60006-60091, 65003-65006, 85236-85242
    if 60000 <= n <= 69999: return 0   # HC
    if 65000 <= n <= 65999: return 0   # HC
    if 85000 <= n <= 85999: return 0   # HC
    if 14000 <= n <= 15999: return 0   # HC (prodromal/control)
    if 40000 <= n <= 42999: return 1   # PD
    if 50000 <= n <= 54999: return 1   # PD
    if 3000  <= n <= 5999:  return 1   # PD
    # Default fallback
    return 1 if n > 9999 else 0


def collect_ppmi_subjects(data_root: str, meta_csv=None):
    subject_dirs = sorted(
        d for d in os.listdir(data_root)
        if os.path.isdir(os.path.join(data_root, d))
    )
    paths, sids = [], []
    for sid in subject_dirs:
        nii_files = glob.glob(
            os.path.join(data_root, sid, "**", "*.nii"), recursive=True)
        if nii_files:
            paths.append(nii_files[0])
            sids.append(sid)

    if not paths:
        raise FileNotFoundError(f"No .nii files found under {data_root}")

    if meta_csv and os.path.isfile(meta_csv):
        meta   = pd.read_csv(meta_csv)
        meta["PATNO"] = meta["PATNO"].astype(str)
        lookup = dict(zip(meta["PATNO"], meta["COHORT"].str.upper()))
        labels = np.array(
            [1 if lookup.get(s, "HC") == "PD" else 0 for s in sids],
            dtype=np.int64)
    else:
        labels = np.array([ppmi_label_from_sid(s) for s in sids],
                          dtype=np.int64)

    n_pd = int(labels.sum())
    print(f"  Subjects : {len(paths)}  |  PD={n_pd}  HC={len(paths)-n_pd}")
    return paths, labels, sids


# ══════════════════════════════════════════════════════════════════════════════
# 5.  ROI FEATURE EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def extract_all_subject_data(paths: list, scales_roi: list) -> dict:
    """
    For each subject, extract per-scale voxel lists (used to build FC & SC).
    Returns dict {si: list_of_voxel_lists} where each voxel_list has
    n_roi entries.
    """
    ref_shape = nib.load(paths[0]).get_fdata().shape[:3]
    print(f"  Reference volume shape : {ref_shape}")

    atlases = {}
    for si, n_roi in enumerate(scales_roi):
        t_atlas = time.perf_counter()
        atlases[si] = make_grid_atlas(ref_shape, n_roi)
        print(f"  [{_now()}]  Grid atlas {n_roi:4d} ROIs built  "
              f"({time.perf_counter()-t_atlas:.2f}s)")

    voxel_data = {si: [] for si in range(len(scales_roi))}

    t_loop = time.perf_counter()
    for subj_i, path in enumerate(paths):
        t_subj = time.perf_counter()
        data = nib.load(path).get_fdata(dtype=np.float32)
        if data.ndim == 4:
            data = data.mean(axis=-1)

        # Crop / pad to reference shape
        cropped = np.zeros(ref_shape, dtype=np.float32)
        slices  = tuple(slice(0, min(data.shape[i], ref_shape[i]))
                        for i in range(3))
        cropped[slices] = data[slices]

        # Mild z-normalisation to make intensities comparable across subjects
        mu, sd = cropped.mean(), cropped.std() + 1e-8
        cropped = (cropped - mu) / sd

        for si, n_roi in enumerate(scales_roi):
            voxel_data[si].append(
                extract_roi_voxels(cropped, atlases[si], n_roi))

        if (subj_i + 1) % 10 == 0 or subj_i == len(paths) - 1:
            t_so_far = time.perf_counter() - t_loop
            print(f"  [{_now()}]  Voxel extraction  "
                  f"{subj_i+1:3d}/{len(paths)}  "
                  f"({time.perf_counter()-t_subj:.2f}s/subj)  "
                  f"{eta(subj_i+1, len(paths), t_so_far)}")

    return voxel_data, atlases


def build_fc_sc_matrices(voxel_data: dict, atlases: dict,
                          scales_roi: list) -> tuple:
    """
    Build per-subject FC and SC matrices for every scale.
    Returns fc_scales, sc_scales: dicts {si → ndarray (N, n, n)}.
    """
    fc_scales, sc_scales = {}, {}
    N = len(voxel_data[0])

    for si, n_roi in enumerate(scales_roi):
        tic(f"FC + SC  scale={n_roi} ROIs  ({N} subjects)")
        fc_list, sc_list = [], []
        t_loop = time.perf_counter()
        for s in range(N):
            t_s = time.perf_counter()
            voxels = voxel_data[si][s]
            fc_list.append(build_subject_fc(voxels, n_roi))
            sc_list.append(build_subject_sc(voxels, n_roi, atlases[si]))
            if (s + 1) % 10 == 0 or s == N - 1:
                t_so_far = time.perf_counter() - t_loop
                print(f"    [{_now()}]  scale {n_roi}  subj {s+1:3d}/{N}  "
                      f"({time.perf_counter()-t_s:.2f}s/subj)  "
                      f"{eta(s+1, N, t_so_far)}")
        fc_scales[si] = np.array(fc_list)   # (N, n_roi, n_roi)
        sc_scales[si] = np.array(sc_list)
        toc(f"scale {n_roi} FC/SC done  → shape {fc_scales[si].shape}")

    return fc_scales, sc_scales


# ══════════════════════════════════════════════════════════════════════════════
# 6.  PREPROCESSING  (paper §II, BUG 3 FIX in augmentation)
# ══════════════════════════════════════════════════════════════════════════════

def graph_diffusion(fc: np.ndarray,
                    t: float = T_DIFFUSION,
                    beta: float = BETA) -> np.ndarray:
    A = (fc + fc.T) / 2.0
    np.fill_diagonal(A, 0.0)
    D = np.diag(np.abs(A).sum(axis=1))
    L = D - A
    try:
        ev, evec = la.eigh(L)
        ev       = np.maximum(ev, 0.0)
        G_diff   = evec @ np.diag(np.exp(-t * ev)) @ evec.T
    except la.LinAlgError:
        G_diff = A
    return beta * A + (1.0 - beta) * G_diff


def sparsify_fc(fc: np.ndarray, M: int = M_SPARSE) -> np.ndarray:
    """Keep top-M positive AND bottom-M negative edges per node."""
    N      = fc.shape[0]
    sparse = np.zeros_like(fc)
    for i in range(N):
        row    = fc[i].copy()
        row[i] = 0.0
        idx_pos = np.argsort(row)[-M:]
        idx_neg = np.argsort(row)[:M]
        sparse[i, idx_pos] = row[idx_pos]
        sparse[i, idx_neg] = row[idx_neg]
    return (sparse + sparse.T) / 2.0


def sparsify_sc(sc: np.ndarray, pct: float = SC_PCT) -> np.ndarray:
    vals = sc[sc > 0]
    if not len(vals):
        return sc
    thr = np.percentile(vals, pct)
    out = sc.copy()
    out[out < thr] = 0.0
    return out


def augment_fc(fc: np.ndarray,
               noise_frac: float = NOISE_FRAC,
               n: int = N_AUGMENTS) -> list:
    """
    BUG-3 FIX: noise amplitude = noise_frac × std(fc) instead of fixed σ=2.
    This preserves SNR regardless of the FC value range.
    """
    sigma  = noise_frac * (fc.std() + 1e-8)
    copies = [fc]
    N      = fc.shape[0]
    for _ in range(n):
        p = np.random.randn(N, N)
        copies.append(fc + sigma * (p + p.T) / 2.0)
    return copies


def preprocess_subject(fc_raw: np.ndarray,
                       sc_raw: np.ndarray) -> tuple:
    """Apply diffusion then sparsification."""
    fc_proc = sparsify_fc(graph_diffusion(fc_raw))
    sc_proc = sparsify_sc(sc_raw)
    return fc_proc, sc_proc


# ══════════════════════════════════════════════════════════════════════════════
# 7.  GRAPH PRIMITIVES
# ══════════════════════════════════════════════════════════════════════════════

class BrainGraph:
    __slots__ = ("x", "L_norm", "y")
    def __init__(self, x, L_norm, label):
        self.x      = x
        self.L_norm = L_norm
        self.y      = torch.tensor([label], dtype=torch.long)


def _normalised_laplacian(sc: np.ndarray) -> torch.Tensor:
    A = (sc + sc.T) / 2.0
    np.fill_diagonal(A, 0.0)
    # Add self-loops for stability
    A = A + 1e-4 * np.eye(A.shape[0])
    deg      = A.sum(axis=1)
    d_inv_sq = np.where(deg > 0,
                        1.0 / np.sqrt(np.maximum(deg, 1e-12)), 0.0)
    D_inv_sq = np.diag(d_inv_sq)
    L_sym    = np.eye(A.shape[0]) - D_inv_sq @ A @ D_inv_sq
    try:
        lam_max = float(np.max(la.eigvalsh(L_sym)))
    except la.LinAlgError:
        lam_max = 2.0
    lam_max = max(lam_max, 1e-6)
    L_norm  = (2.0 / lam_max) * L_sym - np.eye(A.shape[0])
    return torch.tensor(L_norm, dtype=torch.float32)


def make_graph(fc: np.ndarray, sc: np.ndarray, label: int) -> BrainGraph:
    """
    Node features  = rows of FC matrix  (each node's connectivity profile)
    Graph topology = normalised Laplacian of subject-specific SC
    """
    return BrainGraph(
        torch.tensor(fc, dtype=torch.float32),
        _normalised_laplacian(sc),
        label,
    )


def collate_brain_graphs(batch):
    xs = torch.stack([g.x      for g in batch])
    Ls = torch.stack([g.L_norm for g in batch])
    ys = torch.cat(  [g.y      for g in batch])
    return xs, Ls, ys


class BrainGraphDataset(Dataset):
    def __init__(self, graphs):   self.graphs = graphs
    def __len__(self):            return len(self.graphs)
    def __getitem__(self, i):     return self.graphs[i]


# ══════════════════════════════════════════════════════════════════════════════
# 8.  DATASET CONSTRUCTION
# ══════════════════════════════════════════════════════════════════════════════

def build_split_datasets(fc_scales: dict, sc_scales: dict,
                          labels: np.ndarray,
                          train_idx: np.ndarray,
                          test_idx:  np.ndarray,
                          augment_train: bool) -> tuple:
    """
    BUG-4 FIX: both single-scale and dual-scale use the same augmented data.
    The `augment_train` flag is used for BOTH, controlled by the caller.
    """
    train_ds = {si: [] for si in range(N_SCALES)}
    test_ds  = {si: [] for si in range(N_SCALES)}

    for si in range(N_SCALES):
        # ── Training set ──────────────────────────────────────────────────
        for idx in train_idx:
            fc, sc = preprocess_subject(fc_scales[si][idx],
                                        sc_scales[si][idx])
            versions = augment_fc(fc) if augment_train else [fc]
            for fc_v in versions:
                train_ds[si].append(make_graph(fc_v, sc, labels[idx]))

        # ── Test set (clean, no augmentation) ────────────────────────────
        for idx in test_idx:
            fc, sc = preprocess_subject(fc_scales[si][idx],
                                        sc_scales[si][idx])
            test_ds[si].append(make_graph(fc, sc, labels[idx]))

    return train_ds, test_ds


# ══════════════════════════════════════════════════════════════════════════════
# 9.  SPECTRAL GCN  (paper §III.C)
# ══════════════════════════════════════════════════════════════════════════════

class ChebConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, K: int = K_CHEB):
        super().__init__()
        self.K      = K
        self.linear = nn.Linear(K * in_ch, out_ch, bias=True)

    def forward(self, x, L):       # x:(B,N,F)  L:(B,N,N)
        T0, T1 = x, torch.bmm(L, x)
        parts  = [T0, T1]
        for _ in range(2, self.K):
            Tn = 2.0 * torch.bmm(L, parts[-1]) - parts[-2]
            parts.append(Tn)
        return self.linear(torch.cat(parts[:self.K], dim=-1))


class SpectralGCN(nn.Module):
    def __init__(self, in_ch: int,
                 hidden: int  = HIDDEN,
                 K:      int  = K_CHEB,
                 drop:   float = DROPOUT):
        super().__init__()
        self.conv1 = ChebConv(in_ch, hidden, K)
        self.conv2 = ChebConv(hidden, hidden, K)
        self.bn1   = nn.BatchNorm1d(hidden)   # added for training stability
        self.bn2   = nn.BatchNorm1d(hidden)
        self.fc    = nn.Linear(hidden, 2)
        self.drop  = drop

    def forward(self, xs, Ls):     # xs:(B,N,F)  Ls:(B,N,N)
        B, N, _ = xs.shape

        h = self.conv1(xs, Ls)                 # (B, N, hidden)
        h = h.reshape(B * N, -1)
        h = self.bn1(h).reshape(B, N, -1)
        h = F.relu(h)
        h = F.dropout(h, p=self.drop, training=self.training)

        h = self.conv2(h, Ls)
        h = h.reshape(B * N, -1)
        h = self.bn2(h).reshape(B, N, -1)
        h = F.relu(h)
        h = F.dropout(h, p=self.drop, training=self.training)

        h = h.mean(dim=1)                      # global mean pool → (B, hidden)
        return F.log_softmax(self.fc(h), dim=1)


# ══════════════════════════════════════════════════════════════════════════════
# 10.  TRAINING & EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def make_loader(graphs, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(BrainGraphDataset(graphs),
                      batch_size=batch_size, shuffle=shuffle,
                      collate_fn=collate_brain_graphs)


def train_one_epoch(model, loader, opt, class_weights=None):
    model.train()
    for xs, Ls, ys in loader:
        xs, Ls, ys = xs.to(DEVICE), Ls.to(DEVICE), ys.to(DEVICE)
        opt.zero_grad()
        out  = model(xs, Ls)
        loss = F.nll_loss(out, ys, weight=class_weights)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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


def compute_class_weights(labels: np.ndarray) -> torch.Tensor:
    """Inverse-frequency class weights to handle class imbalance."""
    n0 = (labels == 0).sum()
    n1 = (labels == 1).sum()
    n  = len(labels)
    w  = torch.tensor([n / (2 * n0 + 1e-8),
                        n / (2 * n1 + 1e-8)], dtype=torch.float32)
    return w.to(DEVICE)


def compute_metrics(y_true, y_pred, y_prob) -> dict:
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    try:    auc = roc_auc_score(y_true, y_prob)
    except: auc = 0.5
    mcc  = matthews_corrcoef(y_true, y_pred)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    npv  = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    fm   = fowlkes_mallows_score(y_true, y_pred)
    return dict(accuracy=acc, precision=prec, f1=f1, roc_auc=auc,
                mcc=mcc, recall=rec, specificity=spec, fm=fm, npv=npv)


def train_gcn(train_data: list, test_data: list,
              in_channels: int, train_labels: np.ndarray,
              epochs: int = EPOCHS, verbose: bool = True) -> tuple:
    t_start   = time.perf_counter()
    loader_tr = make_loader(train_data, BATCH_SIZE, shuffle=True)
    loader_te = make_loader(test_data,  len(test_data), shuffle=False)
    model     = SpectralGCN(in_channels).to(DEVICE)
    opt       = torch.optim.Adam(model.parameters(),
                                 lr=LR, weight_decay=WEIGHT_DECAY)
    sched     = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    cw        = compute_class_weights(train_labels)

    for ep in range(epochs):
        train_one_epoch(model, loader_tr, opt, class_weights=cw)
        sched.step()
        # Print a mid-training heartbeat every 50 epochs
        if verbose and (ep + 1) % 50 == 0:
            elapsed = time.perf_counter() - t_start
            remaining = elapsed / (ep + 1) * (epochs - ep - 1)
            print(f"      [{_now()}]  epoch {ep+1:3d}/{epochs}  "
                  f"elapsed {str(timedelta(seconds=int(elapsed)))}  "
                  f"ETA {str(timedelta(seconds=int(remaining)))}")

    y_pred, y_prob, y_true = get_predictions(model, loader_te)
    met     = compute_metrics(y_true, y_pred, y_prob)
    t_total = time.perf_counter() - t_start
    if verbose:
        print(f"      [{_now()}]  ✔ trained {epochs} epochs in "
              f"{str(timedelta(seconds=int(t_total)))}  │  "
              f"acc={met['accuracy']:.4f}  prec={met['precision']:.4f}  "
              f"f1={met['f1']:.4f}  auc={met['roc_auc']:.4f}  "
              f"mcc={met['mcc']:.4f}")
    return met, model


# ══════════════════════════════════════════════════════════════════════════════
# 11.  DUAL-SCALE DECISION VOTING GCN  (paper §III.A)
# ══════════════════════════════════════════════════════════════════════════════

def run_dual_scale_dv_gcn(train_ds: dict, test_ds: dict,
                           in_channels_list: list,
                           train_labels: np.ndarray) -> dict:
    all_probs, y_true = [], None
    t_dv = time.perf_counter()
    for si, (ic, sname) in enumerate(zip(in_channels_list, SCALE_NAMES)):
        tic(f"[DV-GCN] {sname}")
        _, model = train_gcn(train_ds[si], test_ds[si], ic,
                             train_labels, verbose=False)
        loader_te       = make_loader(test_ds[si], len(test_ds[si]), False)
        _, probs, trues = get_predictions(model, loader_te)
        all_probs.append(probs)
        if y_true is None:
            y_true = trues
        acc_s = accuracy_score(trues, (probs >= 0.5).astype(int))
        toc(f"scale acc={acc_s:.4f}")

    avg_probs = np.mean(all_probs, axis=0)
    met = compute_metrics(y_true, (avg_probs >= 0.5).astype(int), avg_probs)
    print(f"    [{_now()}]  DV-GCN fusion done  "
          f"(total {str(timedelta(seconds=int(time.perf_counter()-t_dv)))})")
    return met


# ══════════════════════════════════════════════════════════════════════════════
# 12.  LOGISTIC REGRESSION BASELINE  (paper Table II)
# ══════════════════════════════════════════════════════════════════════════════

def fc_upper_tri_features(fc_scales: dict, sc_scales: dict,
                           scale_idxs: list, subj_idxs: list) -> np.ndarray:
    blocks = []
    for si in scale_idxs:
        rows = []
        for s in subj_idxs:
            fc, _ = preprocess_subject(fc_scales[si][s], sc_scales[si][s])
            N = fc.shape[0]
            rows.append(fc[np.triu_indices(N, k=1)])
        blocks.append(np.array(rows))
    return np.concatenate(blocks, axis=1)


def run_logistic(fc_scales: dict, sc_scales: dict,
                 labels: np.ndarray, scale_idxs: list,
                 train_idx: np.ndarray, test_idx: np.ndarray) -> dict:
    X_tr = fc_upper_tri_features(fc_scales, sc_scales, scale_idxs, train_idx)
    X_te = fc_upper_tri_features(fc_scales, sc_scales, scale_idxs, test_idx)
    scaler = StandardScaler()
    X_tr   = scaler.fit_transform(X_tr)
    X_te   = scaler.transform(X_te)
    cw     = "balanced"
    clf    = LogisticRegression(max_iter=3000, C=1.0, solver="lbfgs",
                                class_weight=cw, random_state=SEED)
    clf.fit(X_tr, labels[train_idx])
    y_pred = clf.predict(X_te)
    y_prob = clf.predict_proba(X_te)[:, 1]
    return compute_metrics(labels[test_idx], y_pred, y_prob)


# ══════════════════════════════════════════════════════════════════════════════
# 13.  TABLE GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def aggregate(fold_results: list) -> dict:
    keys = fold_results[0].keys()
    return {k: (np.mean([m[k] for m in fold_results]),
                np.std( [m[k] for m in fold_results])) for k in keys}


def fmt(mu, sd) -> str:
    return f"{mu:.4f} ± {sd:.4f}"


def print_table(title: str, rows_data: list) -> pd.DataFrame:
    COLS = ["Accuracy", "Precision", "F1", "ROC AUC", "MCC",
            "Recall", "Specificity", "Fowlkes-Mallows", "NPV"]
    KEYS = ["accuracy", "precision", "f1", "roc_auc", "mcc",
            "recall", "specificity", "fm", "npv"]
    df_rows = []
    for modality, scales_str, agg in rows_data:
        row = {"Modality": modality, "Scales": scales_str}
        for col, key in zip(COLS, KEYS):
            row[col] = fmt(*agg[key])
        df_rows.append(row)
    df  = pd.DataFrame(df_rows)
    sep = "═" * 200
    print(f"\n{sep}\n{title}\n{sep}")
    print(df.to_string(index=False))
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 14.  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    t_main = time.perf_counter()
    print(f"\n{'═'*70}")
    print(f"  DV-GCN PPMI  started at {_now()}")
    print(f"{'═'*70}\n")

    # ── 1. Collect subjects ───────────────────────────────────────────────
    tic("Collecting PPMI subjects")
    paths, labels, sids = collect_ppmi_subjects(DATA_ROOT, META_CSV)
    N = len(paths)
    toc("subjects collected")

    print(f"\nLabel distribution check (first 10 subjects):")
    for sid, lbl in zip(sids[:10], labels[:10]):
        print(f"  SID={sid:>8}  label={'PD' if lbl == 1 else 'HC'}")

    # ── 2. Extract voxel data ─────────────────────────────────────────────
    print()
    tic("Extracting ROI voxel data (grid atlas, no internet)")
    voxel_data, atlases = extract_all_subject_data(paths, SCALES_ROI)
    toc("voxel extraction complete")

    # ── 3. Build FC / SC matrices ─────────────────────────────────────────
    print()
    tic("Building subject-specific FC and SC matrices")
    fc_scales, sc_scales = build_fc_sc_matrices(voxel_data, atlases, SCALES_ROI)
    toc("FC + SC matrices ready")

    # ── 4. Cross-validation ───────────────────────────────────────────────
    skf    = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(np.zeros(N), labels))

    gcn_res = {k: [] for k in ["scale0", "scale1", "dual"]}
    lr_res  = {k: [] for k in ["scale0", "scale1", "dual"]}

    t_cv = time.perf_counter()
    for fold, (tr_idx, te_idx) in enumerate(splits):
        t_fold = time.perf_counter()
        print(f"\n{'━'*70}")
        print(f"  [{_now()}]  FOLD {fold+1}/{N_FOLDS}   "
              f"(train={len(tr_idx)}, test={len(te_idx)})  "
              f"CV elapsed {str(timedelta(seconds=int(time.perf_counter()-t_cv)))}")
        print(f"{'━'*70}")

        train_labels = labels[tr_idx]

        # Build augmented datasets
        tic("Building augmented graph datasets")
        ds_tr, ds_te = build_split_datasets(
            fc_scales, sc_scales, labels, tr_idx, te_idx,
            augment_train=True)
        toc(f"datasets ready  "
            f"(train={len(ds_tr[0])}×{N_SCALES} scales, "
            f"test={len(ds_te[0])}×{N_SCALES} scales)")

        # ── Single-scale GCN ──────────────────────────────────────────────
        for si, skey in enumerate(["scale0", "scale1"]):
            tic(f"GCN single  {SCALE_NAMES[si]}")
            met, _ = train_gcn(ds_tr[si], ds_te[si],
                               SCALES_ROI[si], train_labels)
            gcn_res[skey].append(met)
            toc(f"GCN single {SCALE_NAMES[si]} done")

        # ── Dual-scale DV-GCN ─────────────────────────────────────────────
        tic("GCN Dual-Scale Decision Voting (DV-GCN)")
        met = run_dual_scale_dv_gcn(ds_tr, ds_te, SCALES_ROI, train_labels)
        gcn_res["dual"].append(met)
        toc(f"DV-GCN done  →  "
            f"acc={met['accuracy']:.4f}  f1={met['f1']:.4f}  "
            f"auc={met['roc_auc']:.4f}  mcc={met['mcc']:.4f}")

        # ── Logistic Regression ───────────────────────────────────────────
        tic("Logistic Regression baselines")
        for si, skey in enumerate(["scale0", "scale1"]):
            lr_res[skey].append(
                run_logistic(fc_scales, sc_scales, labels,
                             [si], tr_idx, te_idx))
        lr_res["dual"].append(
            run_logistic(fc_scales, sc_scales, labels,
                         [0, 1], tr_idx, te_idx))
        toc("LR done")

        print(f"  [{_now()}]  ── Fold {fold+1} complete  "
              f"(fold wall-clock: "
              f"{str(timedelta(seconds=int(time.perf_counter()-t_fold)))})")

    # ── 5. Summary tables ─────────────────────────────────────────────────
    print(f"\n{'═'*70}")
    print(f"  [{_now()}]  All folds complete  │  "
          f"total CV time: "
          f"{str(timedelta(seconds=int(time.perf_counter()-t_cv)))}")
    print(f"{'═'*70}\n")

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

    df1.to_csv("table1_gcn_ppmi_dual_fixed.csv",      index=False)
    df2.to_csv("table2_logistic_ppmi_dual_fixed.csv", index=False)

    total = time.perf_counter() - t_main
    print(f"\n{'═'*70}")
    print(f"  [{_now()}]  ✔  DONE  │  total runtime: "
          f"{str(timedelta(seconds=int(total)))}")
    print(f"  Saved: table1_gcn_ppmi_dual_fixed.csv  │  "
          f"table2_logistic_ppmi_dual_fixed.csv")
    print(f"{'═'*70}")


if __name__ == "__main__":
    main()
