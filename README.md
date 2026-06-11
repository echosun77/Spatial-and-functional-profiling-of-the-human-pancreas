# Functional-spatial mapping of human pancreatic niches using slice-seq

## Author

Yike Xie

## Overview

This repository contains custom scripts used for:

- preprocessing
- differential expression
- downstream analysis

## Data

This project integrates:

- snRNA-seq data
- spatial transcriptomics data
- immunostaining imaging data
- calcium imaging data for functional measurements

## Environment

Analysis was performed using Python 3.12 and R 4.3.

Recommended package management:
- conda

## Requirements

Software used in the workflow:

- pandas
- scanpy
- numpy
- scipy
- memento
- squidpy
- Cell2location
- matplotlib
- seaborn
- adjustText
- sklearn
- rpy2
- R (DESeq2, ggplot2)

## Directory Structure

```text
notebooks_for_manuscript_figures/              scripts for generating manuscript figures

1snRNA-Seq_preprocessing/                      preprocessing pipeline for snRNA-seq data

2snRNA-Seq_processing/                         scripts for snRNA-seq cell type/state annotation

3snRNA-Seq_DEG_analysis_by_memento/            scripts for running memento-based DEG analysis on the local server

4snRNA-Seq_DEG_analysis_by_DESeq/              scripts for running DESeq-based DEG analysis

5spatial_data_deconvolution_by_Cell2location/  scripts for running cell2location deconvolution on spatial transcriptomics data using the Alvis server

6IH_cell_segmentation_annotation/              scripts for immunostaining analysis of pancreatic slices

7extra_islet_beta_mitochondria_staining/       scripts for mitochondrial immunostaining analysis

utils.py                                       Python file containing custom helper functions

