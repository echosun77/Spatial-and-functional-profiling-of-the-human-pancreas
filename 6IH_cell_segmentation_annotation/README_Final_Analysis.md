# Final Analysis of All Donors - Multi-Donor Pancreatic Islet Analysis

## Overview

This notebook (`Final_Analysis_of_All_donors.ipynb`) performs comprehensive statistical analysis and visualization of pancreatic islet cell composition across multiple donors. It integrates cell-level data from individual donor analyses, maps donors to disease groups based on metadata, and generates publication-ready comparative plots examining cell type distributions, islet characteristics, and disease-specific patterns.

## Purpose

The pipeline analyzes:
- **Cell type composition** across donors and disease groups
- **Islet-level characteristics** (size, cell counts, composition)
- **Disease group comparisons** (T1D, T2D, Pre-diabetes, BMI categories)
- **Size-stratified analyses** (small, medium, large islets)
- **Statistical differences** between conditions using Mann-Whitney U tests

## Input Requirements

### Required CSV Files

**Primary input**: Clustered and filtered CSV files from individual donor analyses
- Expected path pattern: `{BASE_DIR}/Donor_*/Nuclei_pos_csv/combined_filtered_merged_clustered_filtered.csv`
- Alternative: Single CSV file specified in `CSV_PATH`

**Metadata file**: Donor disease classification
- Expected: `{BASE_DIR}/human_donor_info_Camunas_Pancreas.csv`
- Required columns:
  - `index` or `Donor ID`: Donor identifier
  - `T1D`: Type 1 diabetes (yes/no)
  - `T2D`: Type 2 diabetes (yes/no)
  - `HbA1c (%)`: Glycated hemoglobin percentage
  - `BMI`: Body mass index

### Required CSV Columns

Cell-level data should contain:
- `roi_name`: Region of interest identifier
- `cluster`: Islet/cluster assignment (>0 for cells in islets)
- `cell_type`: Cell classification (alpha/beta/delta/nuclei/multihormonal)
- `donor` or `donor_folder`: Donor identifier
- `image_stem` (if `donor` is absent): For extracting donor from filename

### Directory Structure
```
BASE_DIR/
├── Donor_F1/
│   └── Nuclei_pos_csv/
│       └── combined_filtered_merged_clustered_filtered.csv
├── Donor_F2/
│   └── Nuclei_pos_csv/
│       └── combined_filtered_merged_clustered_filtered.csv
├── Donor_U7/
│   └── Nuclei_pos_csv/
│       └── combined_filtered_merged_clustered_filtered.csv
├── human_donor_info_Camunas_Pancreas.csv
└── analysis_plots/  (created automatically)
```

## Configuration

### Key Paths
```python
CSV_PATH = Path(r"D:\Kajsa\Dardel\Donor_F2\Nuclei_pos_csv\combined_filtered_merged_clustered_filtered.csv")
BASE_DIR = Path(r"D:\Kajsa\Dardel")
OUT_DIR = BASE_DIR / "analysis_plots"
```

### Column Names
```python
ROI_COL = "roi_name"
CLUSTER_COL = "cluster"
CELLTYPE_COL = "cell_type"
DOM_CELLTYPE_COL = "dominant_cell_type"
IMAGESTEM_COL = "image_stem"
```

### Disease Group Ordering
```python
desired_order = ['T1D', 'T2D', 'Pre-diabetes', 'BMI >30', 'BMI <30']
```

### Cell Type Groups
```python
bar_groups = ['beta', 'alpha', 'delta']  # Main hormone-producing cells
table_groups = ['multihormonal', 'non hormone signal cell']  # Additional categories
```

## Disease Group Classification Logic

Donors are classified into disease groups based on metadata:

1. **T1D**: `T1D` column = 'yes'
2. **T2D**: `T2D` column = 'yes'
3. **Pre-diabetes**: HbA1c ≥ 5.7% (if not T1D/T2D)
4. **BMI >30**: BMI > 30 (obese, non-diabetic)
5. **BMI <30**: BMI ≤ 30 (lean, non-diabetic)
6. **BMI_unknown**: BMI data unavailable

Groups are displayed as:
- **T1D**, **T2D**, **Pre-diabetes**: As-is
- **BMI >30** → **ND - Obese** (Non-Diabetic - Obese)
- **BMI <30** → **ND - Lean** (Non-Diabetic - Lean)

## Output Plots

The pipeline generates the following publication-ready visualizations:

### Plot 1: Islet Counts
- **File**: `01_cells_per_donor_stacked_pct.png`
- **Type**: Stacked bar chart
- **Description**: Cell type composition per donor (alpha, beta, delta)
- **Features**: 
  - Normalized to 100% for hormone-producing cells
  - Shows islet counts and total cell counts above bars
  - Separate table below for multihormonal and non-hormone cells

### Plot 2A: Per-Donor Composition (Ungrouped)
- **File**: `01_cells_per_donor_stacked_pct.png`
- **Type**: Stacked percentage bar chart
- **Description**: Cell type percentages for each donor individually
- **Layout**: Donors shown in order without grouping

### Plot 2B: Per-Donor Composition (Grouped by Disease)
- **File**: `01_cells_per_donor_stacked_pct_by_disease.png`
- **Type**: Stacked percentage bar chart with disease group brackets
- **Description**: Donors grouped by disease condition
- **Features**:
  - Donors within each group sorted by beta cell percentage (low→high)
  - Visual brackets indicating disease groups
  - Gaps between disease groups for clarity

### Plot 2C: Aggregated by Disease Group
- **File**: `02_cells_per_disease_group_stacked_pct.png`
- **Type**: Aggregated stacked bar chart
- **Description**: Combined cell type composition per disease group
- **Calculation**: Pools all cells from all donors in each group

### Plot 3: Islet-Level Boxplots by Cell Type
- **Files**: `04_boxplot_islet_cellcounts_by_condition_{celltype}.png`
- **Type**: Boxplot with KDE overlay (per cell type: alpha, beta, delta)
- **Description**: Distribution of cell type percentages across islets
- **Features**:
  - Each point = one islet
  - Boxplot per disease condition
  - Right-side half-violin KDE curves
  - Pairwise Mann-Whitney U tests with Bonferroni correction
  - Statistical significance markers (*, **, ***, ****)

### Plot 4: Total Cells per Islet by Condition
- **Files**: 
  - `05_boxplot_islet_totalcells_by_condition_FULL.png` (full range)
  - `05_boxplot_islet_totalcells_by_condition_ZOOM_0_400.png` (zoomed 0-400)
- **Type**: Boxplot with KDE overlay
- **Description**: Total cell count per islet across disease conditions
- **Features**:
  - Ignores cell type (total cells only)
  - Two versions: full data range and zoomed view
  - Statistical comparisons between conditions

### Plot 10: Size-Stratified Analysis
- **Files**: `06_boxplot_islet_by_size_and_condition_{celltype}.png`
- **Type**: Multi-panel boxplots
- **Description**: Cell type percentages stratified by islet size (small/medium/large)
- **Panels**: One subplot per islet size category

### Plot 11-13: Size Comparisons
- **Files**: `07_boxplot_compare_sizes_across_conditions_{celltype}.png`
- **Type**: Comparative boxplots
- **Description**: Compares small vs medium vs large islets for each condition
- **Purpose**: Identifies size-dependent composition changes

### Plot 15-16: Condition Comparisons by Size/Bins
- **Files**: Various cell-type and bin-specific plots
- **Type**: Multi-faceted comparative boxplots
- **Description**: Comprehensive comparisons across all variables
- **Dimensions**: Cell type × condition × islet size/bin

## Color Scheme

### Cell Type Colors (Pastel)
```python
colors = {
    'alpha':   (1.00, 0.70, 0.70),  # Light red
    'beta':    (0.70, 0.92, 0.70),  # Light green
    'delta':   (0.80, 0.70, 0.92),  # Light purple
    'non hormone signal cell': (0.75, 0.80, 0.95),  # Light blue
    'multihormonal': (1.00, 0.85, 0.65),  # Light orange
}
```

### Disease Condition Colors
Automatically assigned using matplotlib's `tab10` colormap or custom definitions.

## Statistical Analysis

### Methods
- **Test**: Mann-Whitney U (non-parametric, unpaired)
- **Correction**: Bonferroni (for multiple comparisons)
- **Comparisons**: All pairwise combinations within each plot
- **Significance levels**:
  - `*`: p < 0.05
  - `**`: p < 0.01
  - `***`: p < 0.001
  - `****`: p < 0.0001

### Visualization of Statistics
- Significance brackets above boxplots
- P-values and corrected p-values displayed
- Star notation for quick interpretation

## Key Features

### 1. Robust Donor Identification
- Handles multiple donor naming conventions
- Extracts donor ID from filenames if column is missing
- Normalizes donor names for consistent matching

### 2. Flexible Disease Classification
- Automatic grouping based on metadata
- Falls back gracefully if metadata is unavailable
- Supports custom disease group ordering

### 3. Advanced Visualization
- **Stacked bars**: Normalized percentages for hormone-producing cells
- **Tables**: Non-normalized data for rare cell types (multihormonal, non-hormone)
- **KDE overlays**: Distribution visualization alongside boxplots
- **Color-coded tables**: Heatmap-style coloring for percentages
- **Custom colorbars**: Visual legends for table cell colors

### 4. Data Validation
- Checks for required columns
- Validates cluster assignments (cluster > 0 for islet cells)
- Handles missing data gracefully
- Reports missing CSV files with full paths

## Usage

### Basic Workflow

1. **Set paths**: Update `BASE_DIR` and `CSV_PATH` in cell 2
2. **Run all cells**: Execute notebook from top to bottom
3. **Check output**: All plots saved to `analysis_plots/` directory
4. **Review statistics**: Examine terminal output for p-values

### Testing with Single Donor
To test with a single donor CSV:
```python
CSV_PATH = Path(r"D:\Kajsa\Dardel\Donor_F2\Nuclei_pos_csv\combined_filtered_merged_clustered_filtered.csv")
BASE_DIR = CSV_PATH.parent.parent  # Points to Donor_F2
```

### Multi-Donor Analysis
To analyze all donors:
```python
BASE_DIR = Path(r"D:\Kajsa\Dardel")  # Parent directory containing all Donor_* folders
```

## Plot Customization

### Adjust Figure Size
```python
# In plot_stacked_pct_with_table function
main_height = 4.6      # Main plot height
table_height = 0.75    # Table height (if present)
fig_w = max(10, 0.28 * cur + 3.0)  # Dynamic width based on number of bars
```

### Modify Bar Spacing
```python
LEFT_PAD = 2.8    # Left margin
STEP = 2          # Space between bars
width_frac = 0.78 # Bar width as fraction of STEP
gap_extra = 0.0   # Extra space for gaps between disease groups
```

### Change Label Sizes
```python
seg_fontsize = 12      # Percentage labels inside bars
top_fontsize = 9       # Islet/cell counts above bars
table_fontsize = 10    # X-axis labels below table
```

### Adjust Y-axis Range
```python
Y_LIM_TOP = 120        # Total y-axis limit (allows space above 100%)
Y_ISLETS = 112         # Y position for islet count labels
Y_CELLS = 106          # Y position for cell count labels
```

## Output Summary

All plots are saved as high-resolution PNG files (200 DPI) in the `analysis_plots/` directory:

- **01_** prefix: Per-donor compositions
- **02_** prefix: Aggregated disease group compositions
- **04_** prefix: Islet-level cell type boxplots
- **05_** prefix: Total cells per islet
- **06_** prefix: Size-stratified analyses
- **07_** prefix: Size comparisons
- **Higher numbers**: Advanced multi-dimensional comparisons

## Dependencies

```python
pandas>=2.0
numpy>=1.20
matplotlib>=3.5
scipy>=1.7  # For Mann-Whitney U tests
```

Optional:
```python
natsort  # For natural sorting of filenames
```

## Troubleshooting

### Common Issues

**Problem**: "No donor CSVs found"
- **Solution**: Check that `combined_filtered_merged_clustered_filtered.csv` exists in each `Donor_*/Nuclei_pos_csv/` folder

**Problem**: "Missing both 'donor' and 'image_stem' in CSV"
- **Solution**: Ensure CSV has a donor identifier column or image_stem for extraction

**Problem**: "No islets cells found (cluster > 0)"
- **Solution**: Verify clustering was performed in previous analysis steps; check CLUSTER_COL value

**Problem**: Disease groups not showing
- **Solution**: Verify `human_donor_info_Camunas_Pancreas.csv` exists and has correct columns

**Problem**: Plots look crowded
- **Solution**: Adjust `STEP`, `LEFT_PAD`, and `width_frac` parameters in plotting function

### Data Validation

Check loaded data:
```python
print(f"Total donors: {dfc['donor_base'].nunique()}")
print(f"Total cells: {len(dfc)}")
print(f"Cells in islets: {len(dfc[dfc[CLUSTER_COL] > 0])}")
print(f"Disease groups: {dfc['Disease group'].value_counts()}")
```

## Methods Section for Reports

### Suggested Text

Multi-donor pancreatic islet cell composition data were aggregated from individual donor analyses and merged with clinical metadata. Donors were classified into disease groups based on diabetes diagnosis (T1D, T2D), pre-diabetic status (HbA1c ≥ 5.7%), or BMI categories (>30 kg/m² obese, ≤30 kg/m² lean) for non-diabetic individuals. Cell type distributions were calculated per donor and per islet (cluster_uid). For stacked bar charts, hormone-producing cell types (alpha, beta, delta) were normalized to 100%, while multihormonal and non-hormone-producing cells were displayed separately. Islet-level analyses examined cell type percentages and total cell counts per islet. Size stratification categorized islets as small, medium, or large based on total cell count tertiles. Statistical comparisons between disease groups used two-sided Mann-Whitney U tests with Bonferroni correction for multiple testing. Boxplots display median (line), interquartile range (box), and 1.5× IQR whiskers, with individual islet values shown as jittered points. Kernel density estimation (KDE) overlays visualize distribution shapes using Gaussian kernels with Scott's bandwidth rule. All visualizations were generated using matplotlib (v3.10.6) and saved at 200 DPI resolution.

## Outputs for Publication

### Figure Selection Guide

**Main figures**:
- `01_cells_per_donor_stacked_pct_by_disease.png`: Comprehensive per-donor overview
- `02_cells_per_disease_group_stacked_pct.png`: Disease group summary
- `04_boxplot_islet_cellcounts_by_condition_beta.png`: Key beta cell distribution
- `05_boxplot_islet_totalcells_by_condition_ZOOM_0_400.png`: Islet size comparison

**Supplementary figures**:
- All size-stratified and bin-specific analyses
- Individual cell type boxplots for alpha and delta cells

## Citation

If using this pipeline for published research, cite:
- Statistical methods (Mann-Whitney U test, Bonferroni correction)
- Visualization tools (matplotlib, scipy)
- Original data sources and donor cohort descriptions

## Authors

Developed for multi-donor pancreatic islet comparative analysis.

## Version History

- **Current**: Multi-donor integration with disease classification and statistical testing
- Includes KDE overlays, size stratification, and comprehensive comparative plots

## Contact

For questions about this analysis pipeline, please contact [Your contact information].

---

**Last Updated**: April 2026
