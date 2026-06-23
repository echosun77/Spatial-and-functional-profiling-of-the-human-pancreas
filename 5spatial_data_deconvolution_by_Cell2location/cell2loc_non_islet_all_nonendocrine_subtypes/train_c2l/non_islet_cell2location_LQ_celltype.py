import pandas as pd
import scanpy as sc
import numpy as np
import os
import matplotlib
matplotlib.use("Agg")     # put this before importing pyplot
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ---- numpy 2.x -> 1.x pickle compatibility shim ----
import sys
import importlib

# If numpy._core doesn't exist (NumPy 1.x), alias it to numpy.core
try:
    import numpy._core  # NumPy >=2 provides this
except ModuleNotFoundError:
    import numpy.core as _np_core
    sys.modules['numpy._core'] = _np_core
    # Map common submodules that pickles may reference
    for _name in ('numerictypes', 'multiarray', 'umath', '_multiarray_umath'):
        try:
            sys.modules[f'numpy._core.{_name}'] = getattr(_np_core, _name)
        except Exception:
            pass
# ----------------------------------------------------

print("load relevent folders")
# fig_folder = '../figures/'
res_folder = f'../../results/cell2loc_non_islet/ct_c2l/LQ/'
ref_folder = '../../results/cell2loc_non_islet/ct_ref/'
data_folder = '../../data/'

os.makedirs(ref_folder, exist_ok=True)
os.makedirs(data_folder, exist_ok=True)
os.makedirs(res_folder, exist_ok=True)

print("load spatial and reference datasets")
spatial = sc.read_h5ad(os.path.join(data_folder, 
    'raw_data_c2l_merged_endocrine_score_LiSA_islets_hq.h5ad'))
spatial = spatial[(spatial.obs['sample'] != 'V32') & (spatial.obs['islets_in_out'] != 'in')].copy() 

low_quality_samples = ['C5', 'C6', 'F1', 'P1', 'P2', 'P6', 'U1', 'U3', 
                       'U4', 'U5', 'U6-slice', 'U7', 'V2', 'V5', 'V9', 'V17', 'V23']
print("only work train on low quality samples")
spatial = spatial[spatial.obs['sample'].isin(low_quality_samples)].copy()

print("remove mitochondria-encoded (MT) genes from the spatial dataset")
spatial.var['MT_gene'] = [gene.startswith('MT-') for gene in spatial.var_names]
spatial.obsm['MT'] = spatial[:, spatial.var['MT_gene'].values].X.toarray()
spatial = spatial[:, ~spatial.var['MT_gene'].values]
spat = spatial.copy()

from cell2location.models import RegressionModel, Cell2location
## ===== 1) Import the saved model trained on ref data
print("import the model from ref data")
parse_file = os.path.join(ref_folder, "sn_non_islet_ct_ref.h5ad")
parse_ref = sc.read_h5ad(parse_file)
rm = RegressionModel.load(ref_folder, parse_ref)

## Extracting reference cell types signatures as a pd.DataFrame for spatial mapping
# export estimated expression in each cluster
print("export estimated expression in each cluster")
if 'means_per_cluster_mu_fg' in parse_ref.varm.keys():
    inf_aver = parse_ref.varm['means_per_cluster_mu_fg'][[f'means_per_cluster_mu_fg_{i}'
                                    for i in parse_ref.uns['mod']['factor_names']]].copy()
else:
    inf_aver = parse_ref.var[[f'means_per_cluster_mu_fg_{i}'
                                    for i in parse_ref.uns['mod']['factor_names']]].copy()
inf_aver.columns = parse_ref.uns['mod']['factor_names']

# ===== 2) Harmonize genes (shared only) =====
# cell2location expects the same gene space in ref and spatial
print("select shared genes between spatial and reference data")
shared = np.intersect1d(spat.var_names, inf_aver.index)
spat = spat[:, shared].copy()
inf_aver = inf_aver.loc[shared, :].copy()

# ===== 3) Map spatial beads with Cell2location =====
# Setup spatial anndata (batch_key optional)
print("set up spatial anndata")
Cell2location.setup_anndata(spat, batch_key="sample")

# Initialize from regression model (the signatures we just learned)
print("initialize from regression model")
c2l = Cell2location(
    spat, cell_state_df=inf_aver,
    # expected cell abundance per location
    N_cells_per_location=1.2, # islet cells are denser, exocrine cells suggest to use 1
    # regularization of within-experiment variation in RNA detection sensivity
    detection_alpha=20 # high --20, low --200
    # Slide-seqV2 generally has strong within-slide gradients in capture efficiency
    # Pancreatic islets themselves are small and heterogeneous
    # start with 20, and then increase toward 50-100, use 200 only when want to enforce very strong uniformity
)

# --- knobs you can tweak ---
chunk_epochs = 100          # first pass (your choice)
max_total_epochs = 300     # hard ceiling so we never run forever
window = 10                 # look at the last 50 epochs to judge convergence
rel_tol = 1e-3              # stop if mean relative improvement < 0.1%
abs_tol = 1e-2              # and also if mean absolute change < 1e4 ELBO units

total_trained = 0

def get_elbo_history(c2l):
    """
    Return a 1D numpy array of the training ELBO per epoch.
    Tries a few attribute layouts used by different cell2location/scvi versions.
    """
    # common cases:
    for attr in ("history_", "history"):
        hist = getattr(c2l, attr, None)
        if hist is None:
            continue
        # pandas DataFrame usually with 'elbo_train' or 'elbo'
        for col in ("elbo_train", "elbo", "training_loss"):
            if col in getattr(hist, "columns", []):
                return np.asarray(hist[col].values, dtype=float)
        # sometimes stored as dict of lists
        if isinstance(hist, dict):
            for k in ("elbo_train", "elbo", "training_loss"):
                if k in hist:
                    return np.asarray(hist[k], dtype=float)
    # fallback: empty
    return np.asarray([], dtype=float)

while total_trained < max_total_epochs:
    # train another chunk
    c2l.train(
        max_epochs=chunk_epochs,
        batch_size=2048,      # or None since you have RAM; 2048 is faster per-epoch
        train_size=1,
        lr=1e-3,
        accelerator="gpu",
    )
    total_trained += chunk_epochs

    # fetch history and decide whether to continue
    y = get_elbo_history(c2l)
    if y.size < window + 1:
        # not enough history yet; do another chunk
        continue

    # compute per-epoch improvements (signed diffs)
    diffs = np.diff(y[-(window+1):])  # length = window
    # Use both absolute and relative criteria (ELBO scale can vary a lot)
    mean_abs = float(np.mean(np.abs(diffs)))
    base = np.maximum(1.0, np.mean(np.abs(y[-(window+1):-1])))
    mean_rel = mean_abs / base

    print(f"[ELBO check] last {window} epochs: mean_abs={mean_abs:.3g}, mean_rel={mean_rel:.3g}")

    if (mean_rel < rel_tol) or (mean_abs < abs_tol):
        print(f"Converged (or flat) after ~{total_trained} epochs. Stopping.")
        break

print(f"Trained for {total_trained} epochs total.")

plt.figure()
c2l.plot_history(10)
plt.savefig(f'{res_folder}/detection_non_islet_alpha_20_LQ_ct.png', dpi=300)

# Posterior & export estimated cell abundances per bead
spat = c2l.export_posterior(
    adata=spat,
    sample_kwargs={"num_samples": 200, "batch_size": 256}
        )
    
# Save model
c2l.save(f"{res_folder}", overwrite=True)
spatial_file = f"{res_folder}/spatial_non_islet_after_c2l_LQ_ct.h5ad"
spat.write(spatial_file)

plt.figure()
c2l.plot_QC()
plt.savefig(f'{res_folder}/c2l_non_islet_QC_LQ_ct.png', dpi=300)

