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
data_folder = '../../data/'
res_folder = '../../results/cell2loc_non_islet/ct_ref'

os.makedirs(data_folder, exist_ok=True)
os.makedirs(res_folder, exist_ok=True)

# ===== 0) Imports & config =====
from cell2location.models import RegressionModel

# Save anndata object with results
parse_file = os.path.join(res_folder, 'sn_non_islet_ct_ref.h5ad')
parse_ref = sc.read_h5ad(parse_file)
rm = RegressionModel.load(res_folder, parse_ref)

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
