import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
import pingouin as pg
from statsmodels.sandbox.stats.multicomp import multipletests



def behavior(x,y_behavior,num_per_group,alpha=0.05):
    """
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4008646/
    https://mdpi-res.com/d_attachment/brainsci/brainsci-11-01531/article_deploy/brainsci-11-01531-v2.pdf?version=1637674129
    designed for only three groups
    :x: brain activity: (num_subjects x len(brain_activity))
    :y: behavioral data: (num_subjects x behavioral_data_types)
    :param y: behavioral scores
    :param conditions: list of condition names for each subject
    :param num_per_group: number of subjects per group
    :return: l_x, l_y, l_x2, l_y2
    where l_x is the latent variable for the conditions (brain scores) and
    l_y is the latent variable for the contrasts (behavioral scores)
    """
    num_group_1,num_group_2,num_group_3 = num_per_group[0],num_per_group[1],num_per_group[2]

    def transform(arr):
        centered = arr - np.mean(arr, axis=0)
        sums = np.sum(centered**2, axis=0)
        normalized = centered / np.sqrt(sums)
        return normalized

    x_AD = x[:num_group_1,:]
    y_AD = y_behavior[:num_group_1,:]
    x_PD = x[num_group_1:num_group_1+num_group_2,:]
    y_PD = y_behavior[num_group_1:num_group_1+num_group_2,:]
    x_NC = x[num_group_1+num_group_2:,:]
    y_NC = y_behavior[num_group_1+num_group_2:,:]

    # transform each group data
    x_AD_transform, x_PD_transform, x_NC_transform = transform(x_AD), transform(x_PD), transform(x_NC)
    y_AD_transform, y_PD_transform, y_NC_transform = transform(y_AD), transform(y_PD), transform(y_NC)

    x_transform = np.concatenate((x_AD_transform, x_PD_transform, x_NC_transform), axis=0)
    y_transform = np.concatenate((y_AD_transform, y_PD_transform, y_NC_transform), axis=0)
    x_transform[np.isnan(x_transform)] = 0
    y_transform[np.isnan(y_transform)] = 0

    # cross product of x and y
    r_AD = np.dot(y_AD_transform.T, x_AD_transform)
    r_PD = np.dot(y_PD_transform.T, x_PD_transform)
    r_NC = np.dot(y_NC_transform.T, x_NC_transform)
    r_behavior = np.concatenate((r_AD, r_PD, r_NC), axis=0)
    r_behavior[np.isnan(r_behavior)] = 0
    U,s,V = np.linalg.svd(r_behavior,full_matrices=False)
    s = np.median(s)
    np.random.seed(0)
    original_statistic = s
    num_permutations = 10000
    permuted_statistics = []
    for j in range(num_permutations):
        x = np.random.permutation(x)
        y_behavior = np.random.permutation(y_behavior)
        x_AD = x[:num_group_1,:]
        y_AD = y_behavior[:num_group_1,:]
        x_PD = x[num_group_1:num_group_1+num_group_2,:]
        y_PD = y_behavior[num_group_1:num_group_1+num_group_2,:]
        x_NC = x[num_group_1+num_group_2:,:]
        y_NC = y_behavior[num_group_1+num_group_2:,:]
        x_AD_transform, x_PD_transform, x_NC_transform = transform(x_AD), transform(x_PD), transform(x_NC)
        y_AD_transform, y_PD_transform, y_NC_transform = transform(y_AD), transform(y_PD), transform(y_NC)
        x_transform = np.concatenate((x_AD_transform, x_PD_transform, x_NC_transform), axis=0)
        y_transform = np.concatenate((y_AD_transform, y_PD_transform, y_NC_transform), axis=0)
        x_transform[np.isnan(x_transform)] = 0
        y_transform[np.isnan(y_transform)] = 0
        r_AD = np.dot(y_AD_transform.T, x_AD_transform)
        r_PD = np.dot(y_PD_transform.T, x_PD_transform)
        r_NC = np.dot(y_NC_transform.T, x_NC_transform)
        r_behavior = np.concatenate((r_AD, r_PD, r_NC), axis=0)
        r_behavior[np.isnan(r_behavior)] = 0
        U,s,V = np.linalg.svd(r_behavior,full_matrices=False)
        permuted_data = np.median(s)
        permuted_statistic = permuted_data
        permuted_statistics.append(permuted_statistic)
    p_value = np.mean([statistic >= original_statistic for statistic in permuted_statistics])
    if p_value < alpha:
        print('significant correlation between data types p(',p_value,') < alpha(',alpha,')')
    else:
        print('no significant correlation between data types p(',p_value,') > alpha(',alpha,')')
    return p_value

def fdr(pvals, alpha=0.05, print_results=False):
    """Calculate FDR corrected p-values for a given set of p-values using various methods,
    and optionally print the results to the console.

    Parameters
    ----------
    pvals : array_like
        Array of p-values of the individual tests.
    alpha : float
        Family-wise error rate.
    print_results : bool
        Whether to print the results to the console. Default is False.

    Returns
    -------
    corrected_pvals : dict
        Dictionary of corrected p-values using various methods.
    """
    # Convert p-values to numpy array
    pvals = np.array(pvals)

    # Perform correction with all methods
    methods = ['bonferroni', 'sidak', 'holm-sidak', 'holm', 'simes-hochberg', 'hommel', 'fdr_bh', 'fdr_by', 'fdr_tsbh', 'fdr_tsbky']
    corrected_pvals = {}

    for method in methods:
        _, corrected, _, _ = multipletests(pvals, alpha, method=method)
        corrected_pvals[method] = corrected
        
        # Print the results if print_results is True
        if print_results:
            print(f"{method}: {corrected}")

    return corrected_pvals

