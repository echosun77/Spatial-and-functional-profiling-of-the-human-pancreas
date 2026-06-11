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
# fig_folder = '../figures/cell2loc_non_islet/ct_ref'
data_folder = '../../data/'
res_folder = '../../results/cell2loc_non_islet/ct_ref'

# os.makedirs(fig_folder, exist_ok=True)
os.makedirs(res_folder, exist_ok=True)

print('Load the reference data (i.e., parse snRNA-Seq data in this case)')
parse = sc.read_h5ad(os.path.join(data_folder, 'parse_snRNA_annotated_YK_raw.h5ad'))
parse = parse[(parse.obs['Doublets'] == 'no') & (parse.obs['cell_type'] != 'Endocrine')]

# ===== 0) Imports & config =====
from cell2location.models import RegressionModel

# your endocrine set (adjust to your labels)
ENDO_TYPES = ['Acinar', 'Ductal', 'Endothelial', 'Immune', 'Stellate', 'Schwann']   # (= PP/γ)

REF_CTYPE_COL = "cell_type"   # <-- change if your ref uses a different column
REF_BATCH_COL = 'sample'          # e.g., "donor" if you want to account for ref batch

from cell2location.utils.filtering import filter_genes
selected = filter_genes(parse, cell_count_cutoff=5, 
                        cell_percentage_cutoff2=0.03, 
                        nonz_mean_cutoff=1.12)

# filter the object
parse = parse[:, selected].copy()

parse_ref = parse.copy()

# ===== 1) Train reference signatures on snRNA-seq =====
# Setup anndata for RegressionModel
RegressionModel.setup_anndata(
    parse_ref,
    labels_key=REF_CTYPE_COL,
    batch_key=REF_BATCH_COL
)

rm = RegressionModel(parse_ref)
# view anndata_setup as a sanity check
rm.view_anndata_setup()

# train the model to estimate the reference cell type signatures
rm.train(max_epochs=1000) # you may increase max epochs by checking rm.plot_history() to reach plat

# In this section, we export the estimated cell abundance (summary of the posterior distribution).
parse_ref = rm.export_posterior(
    parse_ref, sample_kwargs={'num_samples': 1000, 'batch_size': 2048}
)

# Save model
rm.save(f"{res_folder}", overwrite=True)

# Save anndata object with results
parse_file = os.path.join(res_folder, 'sn_non_islet_ct_ref.h5ad')
parse_ref.write(parse_file)

plt.figure()
rm.plot_history(10)
plt.savefig(os.path.join(res_folder, 'ref_training_detection_ct.png'), dpi=300, bbox_inches='tight')

parse_ref = rm.export_posterior(
        parse_ref, use_quantiles=True,
        add_to_varm=["q05","q50", "q95", "q0001"],
        sample_kwargs={'batch_size': 2048}
        )

plt.figure()
rm.plot_QC(summary_name="q05")
plt.savefig(os.path.join(res_folder, 'ref_training_QC_ct.png'), dpi=300, bbox_inches='tight')
