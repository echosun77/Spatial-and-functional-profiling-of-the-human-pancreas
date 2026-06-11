# Pancreatic Islet Single-Cell Segmentation Pipeline

## Overview
A Python pipeline for automated single-cell segmentation and mitochondrial 
protein quantification in human pancreatic tissue from large-scale confocal 
images. Compares VDAC1 (mitochondrial mass) and MTCO1 (Complex IV) expression 
between islet beta cells and extra-islet beta cells at single-cell resolution.


## Pipeline Overview

## Requirements
```bash
conda create -n islet_pipeline python=3.11
conda activate islet_pipeline
pip install cellpose tifffile scikit-image scikit-learn scipy 
pip install pandas numpy matplotlib seaborn tifffile
```

## Input Data
- **Format**: Multi-channel TIFF (4 channels)
- **Channels**: CH1=VDAC1 (AF488), CH2=MTCO1 (AF647), 
                CH3=Insulin (AF405), CH4=E-cadherin (AF568)
- **Acquisition**: Zeiss LSM980, 20×, pixel size 0.1235 µm/px

## Pipeline Steps

### 1. Cell Segmentation
```python
# Tiled Cellpose segmentation
# TILE_SIZE=4000px, OVERLAP=400px, DS=1 (full resolution)
# Model: cyto3, diameter=100px, flow_threshold=0.8, cellprob_threshold=-1.0
# Input: E-cadherin (primary) + Insulin (secondary)
# Output: all_cells_raw.csv (76,628 cells)
# Runtime: ~34 minutes on Apple M4 Max (Metal GPU)
```

### 2. Beta Cell Classification
```python
# Otsu threshold on insulin_mean distribution
# Threshold: 14.9 AU
# Beta cells: 3,716 / 76,628 total cells
```

### 3. Islet Detection
```python
# DBSCAN on insulin+ cell coordinates
# eps=200px (~24.7µm), min_samples=10
# 51 clusters detected → 7 ducts excluded (visual inspection)
# Valid islets: 43
```

### 4. Extra-Islet Beta Cell Identification
```python
# Multi-criteria filter:
# - distance from nearest islet > 150 µm
# - area: 1,232-7,787 px² (islet beta cell 10th-95th percentile)
# - spatially isolated
```

## Author
Alana Mullins


readme_path = OUTPUT_DIR / "README.md"
with open(readme_path, 'w') as f:
    f.write(readme_text)
print(f"Saved: {readme_path}")
print(f"Saved: {methods_path}")
print("\nBoth documents ready for use!")
