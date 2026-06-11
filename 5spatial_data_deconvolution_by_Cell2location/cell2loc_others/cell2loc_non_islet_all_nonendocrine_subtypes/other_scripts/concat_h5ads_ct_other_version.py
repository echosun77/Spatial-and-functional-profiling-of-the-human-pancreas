"""
Concatenate per-sample Cell2location AnnData files
--------------------------------------------------
Usage:
  python concat_cell2loc_results.py \
      --input_dir ../../results/cell2loc_non_islet/cst_c2l/ \
      --output_dir ../../results/cell2loc_non_islet/
"""

import argparse
import os
import glob
import scanpy as sc
import anndata as ad

# ----------------------------------------------------------
# Parse command-line arguments
parser = argparse.ArgumentParser(description="Concatenate per-sample Cell2location outputs.")
parser.add_argument("--input_dir", required=True, help="Directory containing per-sample .h5ad files.")
parser.add_argument("--output_dir", required=True, help="Directory to save the merged AnnData.")
args = parser.parse_args()

input_dir = args.input_dir.rstrip("/")  # remove trailing slash
output_dir = args.output_dir.rstrip("/")
os.makedirs(output_dir, exist_ok=True)

# ----------------------------------------------------------
# Find all per-sample .h5ad files
pattern = os.path.join(input_dir, "*/spatial_non_islet_after_c2l_*.h5ad")
files = sorted(glob.glob(pattern))
if len(files) == 0:
    raise FileNotFoundError(f"No .h5ad files found in {input_dir}")

print(f"Found {len(files)} files to merge in {input_dir}")

# ----------------------------------------------------------
# Load all AnnData objects
adatas = [sc.read_h5ad(f) for f in files]

# Check consistency of .obsm keys across samples
common_keys = set(adatas[0].obsm.keys())
for a in adatas[1:]:
    common_keys &= set(a.obsm.keys())
print(f"Common .obsm keys across samples: {list(common_keys)}")

# ----------------------------------------------------------
# Concatenate
print("Merging AnnData objects ...")
adata_merged = ad.concat(adatas, label="sample", join="outer", index_unique=None)

# ----------------------------------------------------------
# Save output
out_file = os.path.join(output_dir, "c2l_merged.h5ad")
adata_merged.write(out_file)
print(f"[OK] Saved merged AnnData: {out_file}")

# Optional sanity check
print(f"Merged shape: {adata_merged.shape}")

