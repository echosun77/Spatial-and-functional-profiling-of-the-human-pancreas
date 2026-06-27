# Spatial and functional mapping of the human pancreas reveals endocrine and exocrine cell states in health and metabolic disease

## Author

Yike Xie

## Overview

This repository contains the custom scripts used for the analysis presented in the manuscript *Functional-spatial mapping of human pancreatic niches using Slice-seq*.

The repository includes workflows for:

- preprocessing
- cell type and cell state annotation
- differential expression analysis
- spatial transcriptomics analysis
- image analysis
- downstream statistical analyses and figure generation

## Data

This project integrates multiple data modalities, including:

- single-nucleus RNA sequencing (snRNA-seq)
- spatial transcriptomics
- immunostaining imaging
- calcium imaging for functional measurements

## Environment

Analyses were performed using:

- Python 3.11 and 3.12
- R 4.3

Recommended package manager:

- conda

## Requirements

### Preprocessing software

- Parse Biosciences Split-pipe pipeline (v1.6.2)
- Curio CurioSeeker pipeline (v3.0.0)

### Main software packages

- Scanpy 1.10.4
- AnnData 0.11.3
- Squidpy 1.6.6
- Cell2location 0.1.4
- scVelo 0.2.5
- Cellpose 4.0.6
- DESeq2 1.40.2
- memento-de 0.1.2
- rpy2 3.5.11
- scipy 1.14.1
- statsmodels 0.14.4
- CaImAn

## Directory structure

```text
notebooks_for_manuscript_figures/
    Jupyter notebooks used to generate manuscript figures.

1snRNA-Seq_preprocessing/
    Preprocessing pipeline for snRNA-seq data.

2snRNA-Seq_processing/
    Cell type and cell state annotation.

3snRNA-Seq_DEG_analysis_by_memento/
    Differential expression analysis using memento.

4snRNA-Seq_DEG_analysis_by_DESeq/
    Differential expression analysis using DESeq2.

5spatial_data_deconvolution_by_Cell2location/
    Cell2location-based deconvolution of spatial transcriptomics data.

6IH_cell_segmentation_annotation/
    Immunostaining image segmentation and annotation.

7extra_islet_beta_mitochondria_staining/
    Analysis of mitochondrial immunostaining.

8endocrine_spot_annotation&_islet_segmentation/
    Endocrine spot annotation and islet segmentation for spatial transcriptomics.

9environment_yml_files/
    Conda environment files.

        DESeq_env.yml
            Environment for scripts in
            4snRNA-Seq_DEG_analysis_by_DESeq/.

        Cell2location_env.yml
            Environment for scripts in
            5spatial_data_deconvolution_by_Cell2location/.

        scanpy_env.yml
            Default environment for the remaining Python analyses.

utils.py
    Custom Python helper functions used throughout the project.
```