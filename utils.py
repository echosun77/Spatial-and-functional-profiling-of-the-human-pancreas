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

def normalizedata(adata, log1p=True):
    """normalize the dataset
    
    Parameters:
    ----------
    adata: scanpy.adata
        scanpy adata object

    Returns:
    -------
    adata: scanpy.adata
        scanpy adata object
    """
    sc.pp.normalize_total(adata, target_sum=1e6)
    if log1p == True:
        sc.pp.log1p(adata, base=2)
    return(adata)

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
    sc.external.pp.bbknn(adata, batch_key=batch_label, neighbors_within_batch=neighbors_within_batch)  
    print('Run UMAP')
    sc.tl.umap(adata)
    print('Find communities')
    sc.tl.leiden(adata, flavor="igraph", n_iterations=2, resolution=resolution, key_added=f"leiden_r{resolution}")
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


def load_ct_palette():    
    ct_colors = {
        'Acinar': (0.8352941176470589, 0.3686274509803922, 0.0), # red
        'Stellate': (0.8705882352941177, 0.5607843137254902, 0.0196078431372549), # orange
        'Ductal': (0.00392156862745098, 0.45098039215686275, 0.6980392156862745), # blue
        'Endothelial': (0.5019607843137255, 0.0, 0.5019607843137255), # gray
        'Endocrine': (0.00784313725490196, 0.6196078431372549, 0.45098039215686275), # monocytes # green
        'Immune': (0.984313725490196, 0.6862745098039216, 0.8941176470588236), # pink
        'Schwann': (0.5019607843137255, 0.5019607843137255, 0.5019607843137255), # black
    }

    return ct_colors

def load_cst_palette():
    cst_colors = {
                 # 'Basal_acinar': (0.7372549019607844, 0.5607843137254902, 0.5607843137254902),
                 'Idling_acinar': (0.7372549019607844, 0.5607843137254902, 0.5607843137254902),
                 'Secretory_acinar': (0.9411764705882353, 0.5019607843137255, 0.5019607843137255),
                 'REG+_acinar': (0.5019607843137255, 0.0, 0.0),
                  
                 'Activated_stellates': (1.0, 0.8941176470588236, 0.7686274509803922),
                 'Quiescent_stellates': (1.0, 0.5490196078431373, 0.0),
                  
                 'Canonical_ductal': (0.6901960784313725, 0.7686274509803922, 0.8705882352941177),
                 'Inflam_ductal_1': (0.0, 0.7490196078431373, 1.0), 
                 'Inflam_ductal_2': (0.2549019607843137, 0.4117647058823529, 0.8823529411764706),
                 'MUC5B+_ductal': (0.0, 0.0, 0.5019607843137255),
                
                 'Endothelial': (0.5019607843137255, 0.0, 0.5019607843137255), 
                 'Arterial_ECs': (0.8666666666666667, 0.6274509803921569, 0.8666666666666667),
                 'Venous_ECs': (0.9333333333333333, 0.5098039215686274, 0.9333333333333333),
                 'Capillary_ECs': (0.5019607843137255, 0.0, 0.5019607843137255),
                 'Lymphatic_ECs': (0.29411764705882354, 0.0, 0.5098039215686274),
                    
                 'α': (0.5607843137254902, 0.7372549019607844, 0.5607843137254902),
                 'β': (0.0, 0.5019607843137255, 0.0),
                 'γ': (0.5647058823529412, 0.9333333333333333, 0.5647058823529412),
                 'δ': (0.3333333333333333, 0.4196078431372549, 0.1843137254901961),
        
                 'Macrophages': (0.8470588235294118, 0.7490196078431373, 0.8470588235294118),
                 'Plasmablasts':  (1.0, 0.7529411764705882, 0.796078431372549),
#         (0.29411764705882354, 0.0, 0.5098039215686274),
                 'T_cells': (0.8588235294117647, 0.4392156862745098, 0.5764705882352941),
                 'Mast_cells': (1.0, 0.0, 1.0),
#       (0.7803921568627451, 0.08235294117647059, 0.5215686274509804),

                 'Schwann': (0.5019607843137255, 0.5019607843137255, 0.5019607843137255)} # grey
    
    return cst_colors

def save_with_comments(df, path, *, sep="\t", index=False, comments=None):
    """
    Write a text table with comment lines (starting with '#') at the top.
    `comments` can be a string, list[str], or dict (key: value).
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # normalize comments
    lines = []
    if comments is None:
        lines = []
    elif isinstance(comments, str):
        lines = [comments]
    elif isinstance(comments, dict):
        for k, v in comments.items():
            lines.append(f"{k}: {v}")
    else:
        lines = list(comments)

    # add a timestamp line (optional)
    lines.insert(0, f"saved: {dt.datetime.now().isoformat(timespec='seconds')}")

    with open(path, "w", encoding="utf-8", newline="") as f:
        for s in lines:
            f.write(f"# {s}\n")
        # now write the table header/body after the comments
        df.to_csv(f, sep=sep, index=index)

def read_with_comments(path, *, sep="\t", **kwargs):
    """Read a file saved above; comment lines (starting with '#') are ignored."""
    return pd.read_csv(path, sep=sep, comment="#", **kwargs)

def break_label(label, max_len=30):
    words = label.split()
    lines = []
    current_line = ""

    for word in words:
        if len(current_line + " " + word) <= max_len:
            if current_line:
                current_line += " " + word
            else:
                current_line = word
        else:
            lines.append(current_line)
            current_line = word

    if current_line:
        lines.append(current_line)

    return '\n'.join(lines)

def pval_to_stars(pval):
    if pval < 0.001:
        return '***'
    elif pval < 0.01:
        return '**'
    elif pval < 0.05:
        return '*'
    else:
        return 'ns'

def clean_axis(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_frame_on(False)