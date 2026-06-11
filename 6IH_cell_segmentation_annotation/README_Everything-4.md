# Pancreatic Islet Tissue Analysis Pipeline

## Overview

This pipeline (`Everything-4.ipynb`) performs comprehensive analysis of pancreatic islet tissue images, including cell segmentation, classification, and quantitative analysis. It processes multi-channel fluorescence microscopy images to identify and characterize different pancreatic cell types (alpha, beta, and delta cells).

## Pipeline Workflow

### 1. Image Preprocessing
- Converts `.ome.tif` files to PNG format
- Splits multi-channel images into individual channels:
  - **Channel 1 (Beta)**: Insulin - Green
  - **Channel 2 (Delta)**: Somatostatin - Yellow
  - **Channel 3 (Alpha)**: Glucagon - Red
  - **Channel 4 (Nuclei)**: DAPI - Blue
- Creates merged RGB images with appropriate color mapping
- Applies percentile-based normalization for contrast enhancement

### 2. GPU Configuration
- Automatically detects and configures GPU for Cellpose if available
- Falls back to CPU if GPU is not available
- Runs diagnostics to verify GPU setup

### 3. Cell Segmentation (Cellpose)
- Uses Cellpose deep learning model for cell and nuclei segmentation
- Processes both merged images and individual nuclei channels
- Configurable parameters:
  - Flow threshold
  - Cell probability threshold
  - Tile normalization block size
  - Number of iterations
- Generates binary masks for detected cells

### 4. Mask Filtering
- Filters segmentation masks based on:
  - Minimum cell area (default: 200 pixels)
  - Nuclei position from CSV files
  - Overlap with reference coordinates
- Removes small artifacts and false positives
- Preserves label integrity throughout filtering

### 5. Cell Classification
- Classifies cells based on cytoplasmic signal intensity:
  - **Alpha cells**: Glucagon-positive (red)
  - **Beta cells**: Insulin-positive (green)
  - **Delta cells**: Somatostatin-positive (purple)
  - **Multi-positive**: Cells expressing multiple hormones
  - **Nuclei only**: Cells without sufficient hormone signal
- Uses Otsu thresholding or custom thresholds for classification
- Excludes cells based on area constraints

### 6. Coordinate Matching
- Matches detected cells with CSV coordinate files
- Supports multiple coordinate systems:
  - Pixel coordinates
  - Normalized coordinates (full image)
  - ROI-based normalization
  - Auto-fit isotropic scaling
- Handles electrode offset corrections
- Validates matches with configurable tolerance

### 7. Quantitative Analysis
Extracts per-cell measurements:
- **Area metrics**:
  - Total cell area
  - Nucleus area
  - Nucleus/cell area fraction
- **Signal intensity**:
  - Mean intensity per channel
  - Fraction of positive pixels (cytoplasm only)
  - Mean signal in positive regions
- **Cell type distribution**
- **Spatial coordinates**

### 8. Visualization
Generates multiple output types:
- **Colored masks**: Cell type color-coded overlays
- **Boundary outlines**: Cell boundaries on original images
- **Coordinate overlays**: CSV coordinates overlaid on images
- **Statistical plots**:
  - Cell type distribution bar charts
  - Area distribution histograms
  - Signal intensity distributions
  - Nucleus size distributions

### 9. Data Export
- Per-ROI CSV files with all measurements
- Combined dataset across all ROIs
- Cell count summaries
- Filtered and sorted master files

## Input Requirements

### Directory Structure
```
BASE_DIR/
├── tif_background/          # Raw .ome.tif files
├── tif_preprocessed/        # Preprocessed TIF files
├── png/                     # Merged PNG images
├── Nuclei png/             # Nuclei channel PNGs
├── Nuclei_pos_csv/         # CSV files with nuclei positions
└── [output directories created automatically]
```

### CSV File Format
Nuclei position CSV files should contain:
- `roi_name`: ROI identifier
- Coordinate columns (x, y positions)
- `nuclei_area`: Nucleus area (for filtering, minimum 100 pixels)
- Optional: electrode offset columns

## Configuration

### Key Parameters

```python
# Processing mode
PROCESS_ALL_IMAGES = True  # Set False to process only first image (testing)

# Segmentation parameters
flow_threshold = 10
cellprob_threshold = -3.0
tile_norm_blocksize = 500
itterations = 5000

# Filtering parameters
MIN_AREA_PX = 200           # Minimum cell area
FRACTION_THRESHOLD = 0.05   # Threshold for positive signal classification
MIN_CYTO_PX = 10           # Minimum cytoplasm pixels

# Channel indices
ALPHA_CH = 2   # Glucagon
BETA_CH = 0    # Insulin
DELTA_CH = 1   # Somatostatin
NUCLEI_CH = 3  # DAPI

# Coordinate matching
overlap_radius_px = 5       # Tolerance for coordinate overlap
ROI_PX = 470               # ROI size for scaling
```

### Base Directory
Update the `BASE_DIR` variable to point to your experiment folder:
```python
BASE_DIR = Path(r"D:\Kajsa\Dardel\Donor_F2")
```

## Output Files

### Generated Directories
- `png_masks/` - Raw segmentation masks
- `nuclei_masks/` - Nuclei segmentation masks
- `png_filtered_masks/` - Filtered cell masks
- `nuclei_filtered_masks/` - Filtered nuclei masks
- `annotations_csv/` - Per-cell measurements
- `annotations_colored_masks/` - Color-coded cell type masks
- `annotations_overlays/` - Overlay visualizations
- `Pictures/` - Statistical plots and figures
- `cordinate_overlap/` - Coordinate matching results

### CSV Output Columns
- `label`: Cell ID
- `area`: Cell area (pixels²)
- `cell_type`: Classification (alpha/beta/delta/multi-positive/nuclei)
- `roi_name`: ROI identifier
- `mean_{channel}_cyto`: Mean cytoplasmic intensity
- `frac_{channel}_cyto`: Fraction of positive pixels
- `signalmean_{channel}_cyto`: Mean intensity in positive regions
- `cell_area_px`, `nucleus_area_px`: Area measurements
- `nucleus_area_frac`: Nucleus/cell area ratio
- Coordinate columns (if matched)

## Dependencies

### Required Packages
```bash
pip install cellpose[all] pandas matplotlib tifffile imageio[ffmpeg] scikit-image pillow numpy scipy natsort tqdm
```

### Specific Requirements
- **cellpose**: Deep learning segmentation
- **tifffile**: Reading .ome.tif files
- **scikit-image**: Image processing and morphology
- **pandas**: Data manipulation
- **matplotlib**: Visualization
- **PIL (Pillow)**: Image I/O
- **PyTorch**: Backend for Cellpose (GPU support)

## Usage

### Basic Workflow
1. **Prepare data**: Place `.ome.tif` files in `tif_background/` directory
2. **Configure paths**: Update `BASE_DIR` in the notebook
3. **Run preprocessing**: Execute cells 1-4 to convert images
4. **Run segmentation**: Execute cell 5 to perform Cellpose segmentation
5. **Filter masks**: Execute cell 6 to filter based on CSV coordinates
6. **Classify cells**: Execute cells 7-9 for annotation and classification
7. **Generate visualizations**: Execute cell 10 for plots and overlays
8. **Export results**: Combined CSV files are automatically saved

### Testing Mode
For testing on a single image:
```python
PROCESS_ALL_IMAGES = False
```

This processes only the first image in each directory.

### GPU vs CPU
The pipeline automatically detects GPU availability. For large datasets, GPU is strongly recommended:
- **GPU**: ~30-60 seconds per image
- **CPU**: Several minutes per image

## Cell Type Color Scheme

- 🔴 **Alpha** (Glucagon): Red (255, 64, 64)
- 🟢 **Beta** (Insulin): Green (50, 205, 50)
- 🟣 **Delta** (Somatostatin): Purple (128, 0, 128)
- 🟠 **Alpha-Beta**: Orange (255, 140, 0)
- 🟣 **Alpha-Delta**: Magenta (255, 0, 255)
- 🔵 **Beta-Delta**: Cyan (0, 255, 255)
- 🟡 **Alpha-Beta-Delta**: Gold (255, 215, 0)
- ⚪ **Nuclei**: Light Blue (153, 153, 230)
- ⚫ **Excluded**: Dark Gray (64, 64, 64)

## Troubleshooting

### Common Issues

**Problem**: Out of memory errors during segmentation
- **Solution**: Reduce `tile_norm_blocksize` or process fewer images at once

**Problem**: No masks generated
- **Solution**: Adjust `cellprob_threshold` (try -6 to 0 for nuclei, -8 to -3 for cells)

**Problem**: Too many false positives
- **Solution**: Increase `flow_threshold` or `MIN_AREA_PX`

**Problem**: CSV coordinates don't match images
- **Solution**: Check `force_y_up` setting and coordinate system mode

**Problem**: Missing columns in output CSV
- **Solution**: Verify channel names match between configuration and data

### Performance Optimization
- Enable GPU for significant speedup
- Use `PROCESS_ALL_IMAGES = False` for testing
- Reduce `itterations` for faster (but less accurate) segmentation
- Process images in batches if memory is limited

## Advanced Features

### Custom Thresholding
Replace Otsu thresholding with custom values:
```python
USE_OTSU = False
ABS_THRESHOLDS = [0.2, 0.2, 0.2, 0.2]  # Per-channel thresholds
```

### Multi-label Classification
Cells expressing multiple hormones are automatically detected and classified (e.g., "alpha-beta", "alpha-delta").

### Coordinate System Flexibility
Multiple coordinate transformation modes supported for matching external coordinate files.

## Methods Section for Reports

### Suggested Text for Methods Section

Multi-channel fluorescence microscopy images (.ome.tif format) were processed using a custom Python pipeline (Python 3.11.13) with tifffile (v2025.9.30) for image reading. Individual channels (insulin/beta, somatostatin/delta, glucagon/alpha, and DAPI/nuclei) were normalized using percentile-based contrast enhancement (90th-99th percentiles) and merged into RGB images with channel-specific color mapping. Cell and nuclei segmentation was performed using Cellpose (v4.0.6) with the generalist model on GPU-accelerated hardware (PyTorch v2.5.1, CUDA 12.1). Segmentation parameters were optimized for pancreatic tissue: flow threshold = 10, cell probability threshold = -3.0, tile normalization block size = 500 pixels, and 5000 iterations. Nuclei-specific segmentation used flow threshold = 0.4 and cell probability threshold = 0. Segmentation masks were filtered to remove artifacts smaller than 200 pixels and validated against reference nuclei positions using spatial overlap criteria (5-pixel tolerance). Cell classification was performed by quantifying cytoplasmic signal intensity (excluding DAPI-thresholded nuclear regions dilated by 1 pixel) using Otsu thresholding per channel. Cells were classified as alpha (glucagon-positive), beta (insulin-positive), or delta (somatostatin-positive) if ≥5% of cytoplasmic pixels exceeded the channel-specific threshold. Multi-positive cells expressing multiple hormones were identified and labeled accordingly. Per-cell measurements included total cell area, nuclear area, nuclear-to-cell area ratio, mean signal intensity per channel, fraction of positive pixels, and spatial coordinates. Quantitative analysis and visualization were performed using scikit-image (v0.25.2), pandas (v2.3.3), and matplotlib (v3.10.6). Cells with areas outside physiological ranges or insufficient cytoplasmic pixels (<10 pixels) were excluded from analysis.

## Citation

If you use this pipeline in your research, please cite the relevant tools:
- **Cellpose**: Stringer, C., Wang, T., Michaelos, M. & Pachitariu, M. Cellpose: a generalist algorithm for cellular segmentation. *Nature Methods* (2021).

## Authors

Developed for pancreatic islet tissue analysis at [Your Institution].

## License

[Specify license if applicable]

## Version History

- **v4.0**: Current version with integrated preprocessing, segmentation, filtering, and analysis
- Includes coordinate matching and comprehensive visualization

## Contact

For questions or issues, please contact [Your contact information].

---

**Last Updated**: April 2026
