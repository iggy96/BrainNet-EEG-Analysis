import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
import pingouin as pg
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

def transform(arr):
    centered = arr - np.mean(arr, axis=0)
    sums = np.sum(centered**2, axis=0)
    normalized = centered / np.sqrt(sums)
    return normalized

x_transform = transform(x)

conditions = ['AD1','AD2', 'AD3', 'PD1', 'PD2', 'PD3', 'NC1', 'NC2', 'NC3']
num_group_1,num_group_2,num_group_3 = 3,3,3

# contrast matrix
contrast_1_a = np.sqrt(0.3/(num_group_1+num_group_2))
contrast_1_a = np.repeat(-1*contrast_1_a,num_group_1+num_group_2)
contrast_1_b = np.sqrt(0.7/(num_group_3))
contrast_1_b = np.repeat(contrast_1_b,num_group_3)
contrast_1 = np.concatenate((contrast_1_a,contrast_1_b))

sanity_check = np.sum(np.square(contrast_1))
print('sanity check for contrasts 1 : {}'.format(sanity_check))
contrast_2_a = np.sqrt(0.5/(num_group_1))
contrast_2_a = np.repeat(-1*contrast_2_a,num_group_1)
contrast_2_b = np.sqrt(0.5/(num_group_2))
contrast_2_b = np.repeat(contrast_2_b,num_group_2)
contrast_2_c = np.repeat(0,num_group_3)
contrast_2 = np.concatenate((contrast_2_a,contrast_2_b,contrast_2_c))
sanity_check = np.sum(np.square(contrast_2))
print('sanity check for contrasts 2 : {}'.format(sanity_check))
y_contrast = np.array([contrast_1,contrast_2]).T

con = np.array([[1,1,0],
                [1,1,0],
                [1,1,0],
                [0,-1,1],
                [0,-1,1],
                [0,-1,1],
                [-1,0,-1],
                [-1,0,-1],
                [-1,0,-1]])


y__contrast = transform(con)
y_contrast = y__contrast

r_contrast = np.dot(y_contrast.T,x_transform)

U,s,V = np.linalg.svd(r_contrast,full_matrices=False)

# transform U and V in accordance with paper
U,V = -1*U, -1*V

l_x = np.dot(x_transform,V.T)
l_y = np.dot(y_contrast,U)


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
result_12 = pg.ttest(group_1_lx[:,1], group_2_lx[:,1],correction=True)
result_13 = pg.ttest(group_1_lx[:,0], group_3_lx[:,0],correction=True)
result_23 = pg.ttest(group_2_lx[:,0], group_3_lx[:,0],correction=True)
pval_12,t_val_12 = result_12['p-val'][0],result_12['T'][0]
pval_13,t_val_13 = result_13['p-val'][0],result_13['T'][0]
pval_23,t_val_23 = result_23['p-val'][0],result_23['T'][0]
pval_123,t_val_123 = np.mean([pval_12,pval_13,pval_23]),np.mean([t_val_12,t_val_13,t_val_23])

# Print the result
print('p-value for group 1 vs group 2 : {}'.format(pval_12),', t-value : {}'.format(t_val_12))
print('p-value for group 1 vs group 3 : {}'.format(pval_13),', t-value : {}'.format(t_val_13))
print('p-value for group 2 vs group 3 : {}'.format(pval_23),', t-value : {}'.format(t_val_23))
print('p-value for group 1 vs group 2 vs group 3 : {}'.format(pval_123),', t-value : {}'.format(t_val_123))



