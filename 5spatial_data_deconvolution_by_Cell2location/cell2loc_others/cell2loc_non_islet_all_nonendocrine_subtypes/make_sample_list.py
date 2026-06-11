# make_sample_list.py
import scanpy as sc, sys, numpy as np
adata_path = sys.argv[1]  # e.g., path to your big spatial AnnData
col = sys.argv[2] if len(sys.argv) > 2 else "sample"

adata = sc.read_h5ad(adata_path)
samples = np.unique(adata.obs[col].astype(str))
with open("samples.txt", "w") as f:
    for s in samples:
        f.write(s + "\n")
print(f"Wrote {len(samples)} samples to samples.txt")

