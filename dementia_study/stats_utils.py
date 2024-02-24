import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
import pingouin as pg
from statsmodels.sandbox.stats.multicomp import multipletests

def contrast_task(x,conditions,num_per_group,show_plot=False):
    """
    Compute the contrast task PLSC for a given contrast matrix
    designed for three groups
    :param x: data matrix (subjects x voxels)
    :param conditions: list of condition names for each subject
    :param num_per_group: number of subjects per group
    :return: l_x, l_y, l_x2, l_y2
    where l_x is the latent variable for the conditions (brain scores) and
    l_y is the latent variable for the contrasts (behavioral scores)
    """
    num_group_1,num_group_2,num_group_3 = num_per_group[0],num_per_group[1],num_per_group[2]

    def transform(arr):
        # Center the elements by subtracting the mean of each column
        centered = arr - np.mean(arr, axis=0)
        
        # Calculate the sum of the squared elements of each column
        sums = np.sum(centered**2, axis=0)
        
        # Divide each element by the square root of the sum of the squared elements of its column
        normalized = centered / np.sqrt(sums)
        
        return normalized

    x_transform = transform(x)

    # contrast matrix
    contrast_1_a = np.repeat(1,num_group_1+num_group_2)
    contrast_1_b = np.repeat(3,num_group_3)
    contrast_1 = np.concatenate((contrast_1_a,contrast_1_b))
    contrast_2_a = np.repeat(1,num_group_1)
    contrast_2_b = np.repeat(-1,num_group_2)
    contrast_2_c = np.repeat(0,num_group_3)
    contrast_2 = np.concatenate((contrast_2_a,contrast_2_b,contrast_2_c))
    y_contrast = np.array([contrast_1,contrast_2]).T
    y_contrast = transform(y_contrast)
    sanity_check_1 = np.sum(np.square(y_contrast[:,0]))
    sanity_check_2 = np.sum(np.square(y_contrast[:,1]))
    print('sanity check for contrasts 1 : {}'.format(sanity_check_1))
    print('sanity check for contrasts 2 : {}'.format(sanity_check_2))


    r_contrast = np.dot(y_contrast.T,x_transform)

    U,s,V = np.linalg.svd(r_contrast,full_matrices=False)

    # transform U and V in accordance with paper
    U,V = -1*U, -1*V

    l_x = np.dot(x,V.T)
    l_y = np.dot(y_contrast,U)

    if show_plot:
        # plot l_x vs conditions
        fig,ax = plt.subplots(1,2,figsize=(10,5))
        ax[0].plot(l_x[:,0],l_x[:,1],'o')
        ax[0].axhline(0, color='black')
        ax[0].axvline(0, color='black')
        ax[0].set_xlabel('l_x1')
        ax[0].set_ylabel('l_x2')
        ax[0].set_title('l_x vs conditions')
        for i, txt in enumerate(conditions):
            ax[0].annotate(txt, (l_x[i,0],l_x[i,1]))
        ax[1].plot(l_y[:,0],l_y[:,1],'o')
        ax[1].axhline(0, color='black')
        ax[1].axvline(0, color='black')
        ax[1].set_xlabel('l_y1')
        ax[1].set_ylabel('l_y2')
        ax[1].set_title('l_y vs conditions')
        for i, txt in enumerate(conditions):
            ax[1].annotate(txt, (l_y[i,0],l_y[i,1]))

    # Conducting two-sample ttest
    group_1_lx = l_x[0:num_group_1,:]
    group_2_lx = l_x[num_group_1:num_group_1+num_group_2,:]
    group_3_lx = l_x[num_group_1+num_group_2:,:]
    result_12 = pg.ttest(group_1_lx[:,1], group_2_lx[:,1])
    result_13 = pg.ttest(group_1_lx[:,0], group_3_lx[:,0])
    result_23 = pg.ttest(group_2_lx[:,0], group_3_lx[:,0])
    pval_12,t_val_12 = result_12['p-val'][0],result_12['T'][0]
    pval_13,t_val_13 = result_13['p-val'][0],result_13['T'][0]
    pval_23,t_val_23 = result_23['p-val'][0],result_23['T'][0]
    pval_123,t_val_123 = np.mean([pval_12,pval_13,pval_23]),np.mean([t_val_12,t_val_13,t_val_23])
    # Print the result using group labels
    print(conditions[0],'vs',conditions[num_group_1],': pval =',pval_12,'t_val =',t_val_12)
    print(conditions[0],'vs',conditions[num_group_1+num_group_2],': pval =',pval_13,'t_val =',t_val_13)
    print(conditions[num_group_1],'vs',conditions[num_group_1+num_group_2],': pval =',pval_23,'t_val =',t_val_23)
    print('mean pval =',pval_123,'mean t_val =',t_val_123)
    print('\n')
    return l_x, l_y

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

def behavior_devv(x, y_behavior, num_per_group, alpha=0.05):
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
    # transform each group data
    x_transform = (x - np.mean(x, axis=0)) / np.std(x, axis=0)
    y_transform = (y_behavior - np.mean(y_behavior, axis=0)) / np.std(y_behavior, axis=0)

    # SVD of cross-product
    U, s, Vt = np.linalg.svd(np.dot(y_transform.T, x_transform), full_matrices=False)
    V = Vt.T

    # Projections
    Lx = np.dot(x_transform, V)
    Ly = np.dot(y_transform, U)

    # Compute original statistic
    original_statistic = np.median(Lx * Ly)

    # Initialize seed for reproducibility
    np.random.seed(42)

    # Define the number of permutations
    num_permutations = 10000

    # Initialize list of permuted statistics
    permuted_statistics = []

    # Permutation test
    for _ in range(num_permutations):
        # Permute only the X matrix while leaving Y unchanged
        x_permuted = np.random.permutation(x_transform)

        # Compute projections for permuted data
        Lx_perm = np.dot(x_permuted, V)
        Ly_perm = np.dot(y_transform, U)

        # Compute permuted statistic (using median)
        permuted_statistic = np.median(Lx_perm * Ly_perm)

        # Append permuted statistic
        permuted_statistics.append(permuted_statistic)

    # Compute p-value
    p_value = np.mean([statistic >= original_statistic for statistic in permuted_statistics])
    if p_value < alpha:
        print('significant correlation between data types p(', p_value, ') < alpha(', alpha, ')')
    else:
        print('no significant correlation between data types p(', p_value, ') > alpha(', alpha, ')')

    return p_value

