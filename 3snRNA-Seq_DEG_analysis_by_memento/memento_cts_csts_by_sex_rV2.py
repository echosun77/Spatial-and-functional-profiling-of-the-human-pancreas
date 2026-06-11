import scanpy as sc
import pandas as pd
import numpy as np
import memento
from scipy.sparse import csr_matrix

# Load raw data
adata = sc.read_h5ad('../data/parse_snRNA_annotated_YK_raw.h5ad')
adata = adata[adata.obs['Doublets'] == 'no'].copy()
adata = adata[adata.obs['sample'] != 'V2'].copy() # remove V2 with BMI 42 -- extreme fat

print('Dataset has been loaded', flush=True)

# Ensure X is in csr sparse format
adata.X = csr_matrix(adata.X)

fea_lists = {
    'cell_type': [
    'Acinar', 
    'Endothelial', 
    'Ductal', 
    'Endocrine',
    'Immune',
    'Stellate',
    'Schwann'
    ],
    'cell_subtype1': [
    'α', 'β', 'δ', #'γ', 
    'Basal_acinar', 'High_enzyme_acinar',
    'Basal_ductal', 'Inflam_ductal_1', 'Inflam_ductal_2', 'MUC5B+_ductal',
    'Capillary_ECs', 'Venous_ECs', #'Arterial_ECs', 'Lymphatic_ECs', 
    'Macrophages', #'T_cells', #'Plasmablasts', 'Mast_cells',
    'Activated_stellates', 'Quiescent_stellates', 
    # 'Schwann'
    ]
    }
ref_group = 'ND-Lean'
capture_rate = 0.15
min_perc_group = 0.1

# cell_n_threshold = {'cell_type': 5, 'cell_subtype': 3}

# Run Memento
for fea in ['cell_type', 'cell_subtype1']:
    print(f'Running {fea} in adata.obs', flush=True)
    print('##########################################', flush=True)

    fea_list = fea_lists[fea]

    # Cell number DataFrame
    # cell_n_df = adata.obs[[fea, 'group', 'Sex']].value_counts().to_frame() >= cell_n_threshold[fea]

    for i, cell_type in enumerate(fea_list):
        print(f'Running {fea}: {cell_type}, {i + 1} in {len(fea_list)} tasks', flush=True)
        print('--------------------------------------', flush=True)

        for group in ['ND-Obese']:
            print(f'Running group: {group}', flush=True)
            print('Subsetting dataset', flush=True)

            # # Fix undefined variable `n_samples`
            # valid_samples = cell_n_df.loc[cell_type].loc[['ND - Lean', group]]
            # samples = valid_samples[valid_samples['count'] == True].index.get_level_values('Sex').unique().tolist()

            # Fix `sample` usage
            adata_tmp = adata[adata.obs['group'].isin([ref_group, group]) 
                              & adata.obs[fea].isin([cell_type]) 
                              # & adata.obs['Sex'].isin(samples)
                              ].copy()

            adata_tmp.obs['group'] = adata_tmp.obs['group'].apply(lambda x: 0 if x == ref_group else 1)

            print('Adding capture_rate and memento_size_factor', flush=True)
            adata_tmp.obs['capture_rate'] = capture_rate

            memento.setup_memento(adata_tmp, q_column='capture_rate')

            print('Creating Memento groups', flush=True)
            memento.create_groups(adata_tmp, label_columns=['group', 'Sex'])

            print(f'Computing 1D moments with min_perc_group={min_perc_group}', flush=True)
            memento.compute_1d_moments(adata_tmp, min_perc_group=min_perc_group)

            # Skip iteration if no genes pass filtering
            if adata_tmp.shape[1] == 0:
                print(f"⚠️ Skipping {cell_type} because no genes passed filtering.", flush=True)
                continue
            else:
                print(f"✅ Processing {cell_type} with {adata_tmp.shape[1]} genes...", flush=True)

                print('Performing 1D hypothesis testing', flush=True)
                sample_meta = memento.get_groups(adata_tmp)
                sample_meta['Sex'] = sample_meta['Sex'].astype('category')

                treatment_df = sample_meta[['group']]
                cov_df = pd.get_dummies(sample_meta['Sex'].astype('category'))

                print('Running memento.ht_1d_moments', flush=True)
                memento.ht_1d_moments(
                    adata=adata_tmp,
                    treatment=treatment_df,
                    covariate=cov_df,
                    num_boot=5000,
                    verbose=1,
                    num_cpus=8  # I have 32 CPUs in Pinbot
                )

                print('Saving results', flush=True)
                # Fix filename formatting
                result_1d = memento.get_1d_ht_result(adata_tmp)
                result_1d.to_csv(
                    f"../tables/by_sex/obese_rV2/{group}_vs_{ref_group}_{fea}_{cell_type}_by_sex.csv",
                    sep=',', index=False
                )
