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
import pandas as pd
import numpy as np

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Concatenate per-sample Cell2location outputs.")
parser.add_argument("--input_dir", required=True, help="Directory containing per-sample .h5ad files.")
parser.add_argument("--output_dir", required=True, help="Directory to save the merged AnnData.")
args = parser.parse_args()

input_dir = args.input_dir.rstrip("/")  # remove trailing slash
output_dir = args.output_dir.rstrip("/")
os.makedirs(output_dir, exist_ok=True)

adata_merged = sc.read_h5ad(os.path.join(output_dir, 'non_islet_c2l_merged_cst.h5ad'))

# Optional sanity check
abund = adata_merged.obsm["q05_cell_abundance_w_sf"].copy() 

abund.to_csv(os.path.join(output_dir, 'q05_cell_abundance_w_sf_non_islet_c2l_cst.csv'), sep=',')
