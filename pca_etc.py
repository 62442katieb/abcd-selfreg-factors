import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
import fancyimpute as fi
import matplotlib.pyplot as plt

fiu_blue = sns.crayon_palette(['Denim'])
sns.set(context='talk', style='ticks')


bx_df = pd.read_csv('imp-dense.csv', header=0, index_col=0)

bx_df['beh_nback_neut-gt-neg_rt'] = bx_df['beh_nback_neutface_cor_rt'] - bx_df['beh_nback_negface_cor_rt']
bx_df['beh_nback_neut-gt-pos_rt'] = bx_df['beh_nback_neutface_cor_rt'] - bx_df['bn_posface_cor_rt']
bx_df['sst_cor_stop_pct'] = 1 - bx_df['bs_incor_stop_percent_total']

bx_vars = ['cash_choice_task', 'sst_cor_stop_pct','upps_y_ss_negative_urgency', 'upps_y_ss_lack_of_planning', 'upps_y_ss_sensation_seeking', 'upps_y_ss_positive_urgency', 'upps_y_lack_perseverance', 'bis_y_ss_bis_sum', 'bis_y_ss_bas_rr', 'bis_y_ss_bas_drive', 'bis_y_ss_bas_fs', 'nihtbx_flanker_agecorrected', 'beh_nback_neut-gt-neg_rt','beh_nback_neut-gt-pos_rt']

impute_pls = fi.SoftImpute(verbose=False)

complete_bx = impute_pls.fit_transform(bx_df[bx_vars])
complete_scaled = scale(complete_bx)

decomp = PCA().fit(complete_scaled)

fig,ax = plt.subplots(figsize=[7,5])
plt.tight_layout(pad=2)
ax2 = ax.twinx()
g = sns.pointplot(np.arange(1,15), decomp.explained_variance_ratio_, markers='x', join=True, ax=ax)
h = sns.lineplot(x=np.arange(1,15), y=np.cumsum(decomp.explained_variance_ratio_), ax=ax2)
h = sns.lineplot(x=np.arange(1,15), y=0.5, ax=ax2, dashes=True, color='0.5')
g.set_xlabel('Number of Components', labelpad=10)
g.set_yticks(np.arange(0.03,0.19, 0.03))
g.set_ylabel('Proportion of Variance Explained', labelpad=10)
ax2.set_ylabel('Total Variance Explained', labelpad=10)
ax2.lines[1].set_linestyle("--")
fig.savefig('scree_plot.png', dpi=300)

comp_df = pd.DataFrame(decomp.components_, index=bx_vars)
comp_df
comp_df.to_csv('pca_factors_loadings.csv')

subj_load = PCA().fit_transform(complete_scaled)

subj_loadings = pd.DataFrame(subj_load[:,0:5], index=bx_df.index, columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5'])
subj_loadings.to_csv('pca_subject_loadings.csv')

np.sum(decomp.explained_variance_ratio_[0:5])
