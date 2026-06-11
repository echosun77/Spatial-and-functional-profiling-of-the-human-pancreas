import pandas as pd
import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings("ignore")

fig_folder = '../figures/'
res_folder = '../results/'
data_folder = '../data/'

os.makedirs(fig_folder, exist_ok=True)
os.makedirs(data_folder, exist_ok=True)
os.makedirs(res_folder, exist_ok=True)

spatial_islet = sc.read_h5ad(os.path.join(data_folder, 'spatial_islet.h5ad'))
parse_islet = sc.read_h5ad(os.path.join(data_folder, 'parse_islet.h5ad'))

from cell2location.models import RegressionModel, Cell2location

## ===== 1) Import the saved model trained on ref data
parse_file = f"{os.path.join(data_folder, 'ref')}/sn_islet_ref.h5ad"
parse_ref = sc.read_h5ad(parse_file)
rm = RegressionModel.load(f"{os.path.join(data_folder, 'ref')}", parse_ref)

## Extracting reference cell types signatures as a pd.DataFrame for spatial mapping
# export estimated expression in each cluster
if 'means_per_cluster_mu_fg' in parse_ref.varm.keys():
    inf_aver = parse_ref.varm['means_per_cluster_mu_fg'][[f'means_per_cluster_mu_fg_{i}'
                                    for i in parse_ref.uns['mod']['factor_names']]].copy()
else:
    inf_aver = parse_ref.var[[f'means_per_cluster_mu_fg_{i}'
                                    for i in parse_ref.uns['mod']['factor_names']]].copy()
inf_aver.columns = parse_ref.uns['mod']['factor_names']

# ===== 2) Harmonize genes (shared only) =====
# cell2location expects the same gene space in ref and spatial
shared = np.intersect1d(spat.var_names, inf_aver.index)
spat = spat[:, shared].copy()
inf_aver = inf_aver.loc[shared, :].copy()

# ===== 3) Map spatial beads with Cell2location =====
# Setup spatial anndata (batch_key optional)
Cell2location.setup_anndata(spat, batch_key="sample")

# Initialize from regression model (the signatures we just learned)
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

# Train mapping model (spot-wise abundances)
c2l.train(
    max_epochs=300,
    batch_size=2048, # don't leave None
)

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