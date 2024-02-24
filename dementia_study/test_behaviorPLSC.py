import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
import pingouin as pg
from scipy.stats import pearsonr
from stats_utils import*

x = np.array([[2,5,6,1,9,1,7,6,2,1,7,3],
              [4,1,5,8,8,7,2,8,6,4,8,2],
              [5,8,7,3,7,1,7,4,5,1,4,3],
              [3,3,7,6,1,1,10,2,2,1,7,4],
              [2,3,8,7,1,6,9,1,8,8,1,6],
              [1,7,3,1,1,3,1,8,1,3,9,5],
              [9,0,7,1,8,7,4,2,3,6,2,7],
              [8,0,6,5,9,7,4,4,2,10,3,8],
              [7,7,4,5,7,6,7,6,5,4,8,8]])

y_behavior = np.array([[15,600],
                      [19,520],
                      [18,545],
                      [22,426],
                      [21,404],
                      [23,411],
                      [29,326],
                      [30,309],
                      [30,303]])


conditions = ['AD1','AD2', 'AD3', 'PD1', 'PD2', 'PD3', 'NC1', 'NC2', 'NC3']
num_group_1,num_group_2,num_group_3 = 3,3,3


def transform(arr):
    centered = arr - np.nanmean(arr, axis=0)
    sums = np.nansum(centered**2, axis=0)
    normalized = np.divide(centered, np.sqrt(sums))
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

# REPLACE NAN WITH MEAN
x_transform[np.isnan(x_transform)] = np.nanmean(x_transform)
y_transform[np.isnan(y_transform)] = np.nanmean(y_transform)
#x_transform[np.isnan(x_transform)] = 0
#y_transform[np.isnan(y_transform)] = 0

print("x_transform: ", x_transform)
print("y_transform: ", y_transform)


# cross product of x and y
r_AD = np.dot(y_AD_transform.T, x_AD_transform)
r_PD = np.dot(y_PD_transform.T, x_PD_transform)
r_NC = np.dot(y_NC_transform.T, x_NC_transform)
r_behavior = np.concatenate((r_AD, r_PD, r_NC), axis=0)
r_behavior[np.isnan(r_behavior)] = np.nanmean(r_behavior)
#r_behavior[np.isnan(r_behavior)] = 0
print("r_behavior: ", r_behavior)

U,s,V = np.linalg.svd(r_behavior,full_matrices=False)
print("U: ", U)
print("s: ", s)
print("V: ", V)
s = np.median(s)

alpha=0.05
np.random.seed(42)
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
    print('-significant correlation between data types p(',p_value,') < alpha(',alpha,')')
else:
    print('-no significant correlation between data types p(',p_value,') > alpha(',alpha,')')






#%%
import numpy as np
from scipy.linalg import svd

# Define the data
x = np.array([[2,5,6,1,9,1,7,6,2,1,7,3],
              [4,1,5,8,8,7,2,8,6,4,8,2],
              [5,8,7,3,7,1,7,4,5,1,4,3],
              [3,3,7,6,1,1,10,2,2,1,7,4],
              [2,3,8,7,1,6,9,1,8,8,1,6],
              [1,7,3,1,1,3,1,8,1,3,9,5],
              [9,0,7,1,8,7,4,2,3,6,2,7],
              [8,0,6,5,9,7,4,4,2,10,3,8],
              [7,7,4,5,7,6,7,6,5,4,8,8]])

y_behavior = np.array([[15,600],
                      [19,520],
                      [18,545],
                      [22,426],
                      [21,404],
                      [23,411],
                      [29,326],
                      [30,309],
                      [30,303]])

# Normalize X and Y
x_transform = (x - np.mean(x, axis=0)) / np.std(x, axis=0)
y_transform = (y_behavior - np.mean(y_behavior, axis=0)) / np.std(y_behavior, axis=0)

# SVD of cross-product
U, s, Vt = svd(np.dot(y_transform.T, x_transform), full_matrices=False)
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
p_value = np.sum(np.array(permuted_statistics) >= original_statistic) / num_permutations

# Print p-value
print("p-value: ", p_value)





