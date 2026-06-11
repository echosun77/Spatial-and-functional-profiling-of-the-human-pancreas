import scanpy as sc
import pandas as pd
import numpy as np
import memento
from scipy.sparse import csr_matrix

# Load raw data
adata = sc.read_h5ad('../data/parse_snRNA_annotated_YK_raw.h5ad')
print('Dataset has been loaded', flush=True)

# Ensure X is in csr sparse format
adata.X = csr_matrix(adata.X)

fea_lists = {
    'cell_type': [
    # 'Acinar', 'Endothelial', 'Ductal', 'Endocrine'
    ],
    'cell_subtype': [
    # 'α', 'β', 
    # 'γ', 
    'δ', 
    'Acinar_0', 'Acinar_1', 'Acinar_2', 
    'Ductal_0', 'Ductal_1', 'Ductal_2'
    ]
}

cell_n_threshold = {'cell_type': 5, 'cell_subtype': 3}

# Run Memento
for fea in ['cell_type', 'cell_subtype']:
    print(f'Running {fea} in adata.obs', flush=True)
    print('##########################################', flush=True)

    fea_list = fea_lists[fea]

    # Cell number DataFrame
    cell_n_df = adata.obs[[fea, 'group', 'sample']].value_counts().to_frame() >= cell_n_threshold[fea]

    for i, cell_type in enumerate(fea_list):
        print(f'Running {fea}: {cell_type}, {i + 1} in {len(fea_list)} tasks', flush=True)
        print('--------------------------------------', flush=True)

        for group in ['ND - Obese', 'T2D']:
            print(f'Running group: {group}', flush=True)
            print('Subsetting dataset', flush=True)

            ## Fix undefined variable `n_samples`
            #valid_samples = cell_n_df.loc[cell_type].loc[['ND - Lean', group]]
            #samples = valid_samples[valid_samples['count'] == True].index.get_level_values('sample').unique().tolist()

            # Fix `sample` usage
            adata_tmp = adata[adata.obs['group'].isin(['ND - Lean', group]) &
                              adata.obs[fea].isin([cell_type]) &
                              # adata.obs['sample'].isin(samples)
                              ].copy()

            adata_tmp.obs['group'] = adata_tmp.obs['group'].apply(lambda x: 0 if x == 'ND - Lean' else 1)

            print('Adding capture_rate and memento_size_factor', flush=True)
            adata_tmp.obs['capture_rate'] = 0.15

            memento.setup_memento(adata_tmp, q_column='capture_rate')

            print('Creating Memento groups', flush=True)
            memento.create_groups(adata_tmp, label_columns=['group', 'sample', 'Sex'])

            print('Computing 1D moments with min_perc_group=0.7', flush=True)
            memento.compute_1d_moments(adata_tmp, min_perc_group=0.7)

            # Skip iteration if no genes pass filtering
            if adata_tmp.shape[1] == 0:
                print(f"⚠️ Skipping {cell_type} because no genes passed filtering.", flush=True)
                continue
            else:
                print(f"✅ Processing {cell_type} with {adata_tmp.shape[1]} genes...", flush=True)

                print('Performing 1D hypothesis testing', flush=True)
                sample_meta = memento.get_groups(adata_tmp)
                sample_meta['sample'] = sample_meta['sample'].astype('category')
                sample_meta['Sex'] = sample_meta['Sex'].astype('category')

                treatment_df = sample_meta[['group']]
                cov_df = pd.get_dummies(sample_meta[['sample', 'Sex']].astype('category'))

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
                    f"../tables/by_sample_sex/memento_ht_1d_moments_{group}_{fea}_{cell_type.replace(' ', '')}_by_sample_sex.csv",
                    sep=',', index=False
                )
