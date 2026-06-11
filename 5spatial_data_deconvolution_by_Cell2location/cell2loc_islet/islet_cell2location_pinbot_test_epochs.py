import pandas as pd
import scanpy as sc
import numpy as np
import os
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

print("load relevent folders")
fig_folder = '../figures/'
res_folder = '../results/'
data_folder = '../data/test_epochs'

os.makedirs(fig_folder, exist_ok=True)
os.makedirs(data_folder, exist_ok=True)
os.makedirs(res_folder, exist_ok=True)

print("load spatial and reference datasets")
spatial_islet = sc.read_h5ad(os.path.join(data_folder, 'spatial_islet.h5ad'))
parse_islet = sc.read_h5ad(os.path.join(data_folder, 'parse_islet.h5ad'))

print("remove mitochondria-encoded (MT) genes from the spatial dataset")
spatial_islet.var['MT_gene'] = [gene.startswith('MT-') for gene in spatial_islet.var_names]
spatial_islet.obsm['MT'] = spatial_islet[:, spatial_islet.var['MT_gene'].values].X.toarray()
spatial_islet = spatial_islet[:, ~spatial_islet.var['MT_gene'].values]
spat = spatial_islet.copy()

from cell2location.models import RegressionModel, Cell2location
## ===== 1) Import the saved model trained on ref data
print("import the model from ref data")
parse_file = f"{os.path.join(data_folder, 'ref')}/sn_islet_ref.h5ad"
parse_ref = sc.read_h5ad(parse_file)
rm = RegressionModel.load(f"{os.path.join(data_folder, 'ref')}", parse_ref)

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
    N_cells_per_location=1.5, # islet cells are denser, exocrine cells suggest to use 1
    # regularization of within-experiment variation in RNA detection sensivity
    detection_alpha=20 # high --20, low --200
    # Slide-seqV2 generally has strong within-slide gradients in capture efficiency
    # Pancreatic islets themselves are small and heterogeneous
    # start with 20, and then increase toward 50-100, use 200 only when want to enforce very strong uniformity
)

# --- knobs you can tweak ---
chunk_epochs = 300          # first pass (your choice)
max_total_epochs = 2000     # hard ceiling so we never run forever
window = 50                 # look at the last 50 epochs to judge convergence
rel_tol = 1e-3              # stop if mean relative improvement < 0.1%
abs_tol = 1e4               # and also if mean absolute change < 1e4 ELBO units

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
        lr=1e-3
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
plt.savefig('detection_alpha_20.png')

# Posterior & export estimated cell abundances per bead
spat = c2l.export_posterior(
    adata=spat,
    sample_kwargs={"num_samples": 1000, "batch_size": 4096}
)

# Save model
c2l.save(f"{res_folder}", overwrite=True)
spatial_file = f"{res_folder}/spatial_islet.h5ad"
spat.write(spatial_file)
