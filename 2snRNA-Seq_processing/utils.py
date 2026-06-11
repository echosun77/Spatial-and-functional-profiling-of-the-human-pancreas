import scanpy as sc
import anndata as ad


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import os
import glob
import datetime


current_time = f"{datetime.datetime.now():%Y-%m-%d %H:%M}"

def log(text):
    return print(f"{datetime.datetime.now():%Y-%m-%d %H:%M}\t" + text)

def return_incompatible(adata, batch, size):
    incompatible_list = []
    for batch_element in np.unique(adata.obs[batch]):
        if(adata[adata.obs[batch] == batch_element].n_obs < size):
            incompatible_list.append(batch_element)
    return incompatible_list

def load_subcluster_dict(folder_path):
    # Work in progress
    
    subcluster_paths = glob.glob(folder_path)
    subcluster_names = [subcluster_path.replace(folder_path, '').replace('.h5ad', '') for subcluster_path in subcluster_paths]

    subcluster_adata = {}
    for i, subcluster_path in enumerate(subcluster_paths):
        subcluster_adata[subcluster_names[i]] = sc.read_h5ad(subcluster_path)
        
    return subcluster_adata

def dimreduc(adata, resolution = 0.3, batch_label = 'batch', neighbors_within_batch = 3): 
    if incompatible := return_incompatible(adata, batch_label, neighbors_within_batch):
        print(f'There is not enough cells within {incompatible}, continuing without these batch elements')
        adata = adata[~adata.obs[batch_label].isin(incompatible)]
    print('Normalize')
    sc.pp.normalize_total(adata)
    print('Scale')
    sc.pp.log1p(adata)
    print('Extract variable genes')
    sc.pp.highly_variable_genes(adata, n_top_genes=2000) 
    print('PCA')
    sc.tl.pca(adata)
    print('Infer neighbors')
    sc.pp.neighbors(adata)
    sc.external.pp.bbknn(adata, batch_key=batch_label, neighbors_within_batch = neighbors_within_batch)  
    print('Run UMAP')
    sc.tl.umap(adata)
    print('Find communities')
    sc.tl.leiden(adata, flavor="igraph", n_iterations=2, resolution = resolution)
    print('finished')   


def proportion_stackedbar(adata, y = 'annotation', x = 'group', sample = 'sample', title = '', color = None, ax = None, xlabel = None):
    sample_size_per_group = adata.obs.groupby([x])[sample].nunique()
    sizes = adata.obs.groupby([y, x]).size()
    props = sizes.groupby(level=1).apply(lambda x: 100 * (x / x.sum()))
    props = props.unstack(level = 1)
    props.index = props.index.droplevel(0)

    new_index = []
    for i, group in enumerate(props.index.astype(str)):
        new_index.append(f'{group} ({sample_size_per_group[i]})')

    props.index = new_index

    ax = props.plot(kind = "bar", stacked=True, ax=ax, width=0.9, color = color)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(0,100)
    ax.legend(title = y, loc = 'upper left', bbox_to_anchor=(1, 1),
            framealpha=0.0, handlelength=0.5, labelspacing = 0.1,
            alignment = 'left', prop={'size': 16}, title_fontsize = 16, handletextpad = 0.2)
    if(xlabel != None):
        ax.set_xlabel(xlabel, size = 20)
    else:
        ax.set_xlabel(x, size = 20)
    ax.set_ylabel('Proportion (%)', size = 20)
    ax.grid(False, axis = 'x')
    ax.grid(True, color = 'gray', axis = 'y')
    plt.title(title)
    plt.xticks(rotation=45, ha = 'right', rotation_mode='anchor', fontsize = 14)