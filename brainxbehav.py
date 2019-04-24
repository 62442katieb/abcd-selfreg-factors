import numpy as np
import pandas as pd
import seaborn as sns
from os.path import join
from nilearn.mass_univariate import permuted_ols
from scipy.stats import pearsonr, spearmanr

#Li & Ji (2005) method for multiple comparisons corrections
#calculating number of effective comparisons M_eff
def jili_sidak_mc(data, alpha):
    import math
    import numpy as np

    mc_corrmat = data.corr()
    eigvals, eigvecs = np.linalg.eig(mc_corrmat)

    M_eff = 0
    for eigval in eigvals:
        if abs(eigval) >= 0:
            if abs(eigval) >= 1:
                M_eff += 1
            else:
                M_eff += abs(eigval) - math.floor(abs(eigval))
        else:
            M_eff += 0
    print('Number of effective comparisons: {0}'.format(M_eff))

    #and now applying M_eff to the Sidak procedure
    sidak_p = 1 - (1 - alpha)**(1/M_eff)
    if sidak_p < 0.00001:
        print('Critical value of {:.3f}'.format(alpha),'becomes {:2e} after corrections'.format(sidak_p))
    else:
        print('Critical value of {:.3f}'.format(alpha),'becomes {:.6f} after corrections'.format(sidak_p))
    return sidak_p, M_eff

data_dir = '/Users/Katie/Dropbox/Projects/abcd-impulsivity'

sst_df = pd.read_csv(join(data_dir, 'corstop_gt_incorstop-aparcaseg.csv'), index_col=0, sep='\t')
sst_df.head()
pc_df = pd.read_csv(join(data_dir, 'pca_subject_loadings.csv'), index_col=0, header=0)
subj_id = pd.read_csv(join(data_dir, 'subjectkey.csv'), index_col=0)

demo_df = pd.DataFrame({'age': sst_df['interview_age'].values,'gender': sst_df['gender'].values}, index=sst_df.index)
#demo_df.head()
sst_df = sst_df.drop(['gender', 'interview_age'], axis=1)
sst_df.sort_index(inplace=True)
pc_df.sort_index(inplace=True)

for key in sst_df.keys():
    sst_df[key] = pd.to_numeric(sst_df[key])

mean_betas = sst_df.mean(axis=0)
mean_betas.to_csv(join(data_dir, 'mean_betas-corstop_gt_incorstop.csv'))

all = pd.concat([pc_df, sst_df, demo_df], sort=False, axis=1)
all.dropna(inplace=True)
all.keys()

sleep_df = pd.read_csv(join(data_dir, 'sleep.csv'), index_col=0)

all = pd.concat([all, sleep_df], sort=False, axis=1)
all.dropna(inplace=True)
all.index.shape

pcs = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5']
demos = ['age', 'gender']
all = all.replace({'M':0, 'F':1})
brain = list(sst_df.keys())

corstop_pc = permuted_ols(all[pcs].values, all[brain].values, all[demos].values, model_intercept=True, two_sided_test=True)
corstop_dem = permuted_ols(all[demos].values, all[brain].values, model_intercept=True, two_sided_test=True)
corstop_sleep = permuted_ols(all['sleep_disturbance_total'].values, all[brain].values, all[demos].values, model_intercept=True, two_sided_test=True)
sig_df = pd.DataFrame(index=brain)

pc_x_sleep = {}

for i in np.arange(0,5):
    pc_x_sleep['PC{0}'.format(i+1)] = spearmanr(all['PC{0}'.format(i+1)],all['sleep_disturbance_total'])
    if np.max(corstop_pc[0][i]) >= 1:
        sig_df['PC{0} nlog pval'.format(i+1)] = corstop_pc[0][i]
        sig_df['PC{0} tscore'.format(i+1)] = corstop_pc[1][i]

sig_df['Age nlog pval'] = corstop_dem[0][0]
sig_df['Age tscore'] = corstop_dem[1][0]
sig_df['Gender nlog pval'] = corstop_dem[0][1]
sig_df['Gender tscore'] = corstop_dem[1][1]

sig_df['Sleep nlog pval'] = corstop_sleep[0][0]
sig_df['Sleep tscore'] = corstop_sleep[1][0]

sig_df.to_csv(join(data_dir, 'brainXbhav.csv'))
pcxsleep_df = pd.DataFrame.from_dict(pc_x_sleep).T

pcxsleep_df.rename({0:'r', 1:'p'}, axis=1, inplace=True)
pcxsleep_df.to_csv('pc_scorr_with_sleep.csv')

jili_sidak_mc(all[pcs], 0.05)
