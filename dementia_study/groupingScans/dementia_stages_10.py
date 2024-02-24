"""
Joshua Ighalo 12/06/2022

dementia study scans sorter

this script sorts the scans into the following categories:
1. severe dementia
2. moderate dementia
3. mild dementia
4. no dementia

Input: .xlsx file with the following columns: scan ID and MMSE
       
"""
import sys
sys.path.insert(1, '/Users/joshuaighalo/Documents/GitHub/eegDementia')
from eeg_libs import *

search_path = r'/Users/joshuaighalo/Downloads/brainNet_datasets/laurel_place/cleaned_dataset/'
df_1 = pd.read_excel (search_path +'lp_clinical.xlsx')
df_2 = pd.read_excel(search_path + 'Scan_details.xlsx')
data_1 = (df_1[['ID','MMSE']]).to_numpy()
data_2 = (df_2[['Baseline','4 months','8 months']]).to_numpy()
ND_MMSE,MILD_MMSE,MOD_MMSE,SEV_MMSE = 25,[20,24],[13,19],12

#   remove scans of participants who died
died = [152,280,409]
idx_died = np.where(np.isin(data_1[:,0], died))[0]
data_1a = np.delete(data_1,idx_died,axis=0)
data_1 = np.delete(data_1,idx_died,axis=0)
data_2 = np.delete(data_2,idx_died,axis=0)


#   seperate scan IDs for each dementia type
def oneFourDigit(x):
    return '{:04d}'.format(x)

arr = []
for i in range(len(data_1)):
    arr.append(oneFourDigit(data_1[:,0][i]))
data_10 = np.array(arr)
data_1 = np.vstack((data_10,data_1[:,1])).T

#   create an array form of the timepoints csv
#   use dates from scan details to sort scans into timepoints 
#   transform date from yy-mm-dd to ddmmyyyy to match filing system of scans
data_2[np.isnan(data_2)] = 0    # replace nan with unused date

def ext(data):
    char = datetime.strptime(str(data)[0:10], '%Y-%m-%d').strftime('%d%m%Y')
    return char

tp_baseline = []
tp_4months = []
tp_8months = []
for i in range(len(data_2)):
    tp_baseline.append(ext(data_2[i,0]))
    tp_4months.append(ext(data_2[i,1]))
    tp_8months.append(ext(data_2[i,2]))
tp_baseline = np.array(tp_baseline)
tp_4months = np.array(tp_4months)
tp_8months = np.array(tp_8months)

#   merge data_1 and timepoints
def stringMerge(x,y):
    return x + y
dataBase = []
data4months = []
data8months = []
for i in range(len(data_1)):
    dataBase.append(stringMerge(data_1[:,0][i], tp_baseline[i]))
    data4months.append(stringMerge(data_1[:,0][i], tp_4months[i]))
    data8months.append(stringMerge(data_1[:,0][i], tp_8months[i]))

#   import dataset scan from directory

root, dirs, files = next(os.walk(search_path), ([],[],[]))
scans = np.array(dirs)

#   remove "_runNumber_" from scan names and remove last five 
#   characters from scan names to get ID
def kill_char(string, indices): 
    #   e.g., data, indexes = "Welcome", {1, 3, 5}
    data, indexes = string,indices
    new_str = "".join([char for idx, char in enumerate(data) if idx not in indexes])
    new_str = new_str[:-5]
    return new_str

dataScans = []
for i in range(len(scans)):
    dataScans.append(kill_char(scans[i],{4,5,6}))
dataScans = np.array(dataScans)

#   baseline scans in the dataset and their positions in the dataset
#   Note: index positions in dataScans = index positions in scans
idx_B = np.where(np.isin(dataScans, dataBase))[0]
dataScans_B = dataScans[idx_B]
idx_4 = np.where(np.isin(dataScans, data4months))[0]
dataScans_4 = dataScans[idx_4]
idx_8 = np.where(np.isin(dataScans, data8months))[0]
dataScans_8 = dataScans[idx_8]

"""
-   Apply mmse threshold as defined by alz.org to classify participants into dementia classes
-   len and indices of data_1a == len and indices of data_1
-   no dementia (ND)
- Based on Folstein, Folstein, McHugh, and Fanjiang. (2001) 
"""
#   no dementia (ND)
idx_ND = np.where(data_1a[:,1] >= ND_MMSE)[0]
data_ND = data_1[:,0][idx_ND]

#   mild dementia (MILD)
idx_MILD = np.where(np.logical_and(data_1a[:,1]>=MILD_MMSE[0],data_1a[:,1]<=MILD_MMSE[1]))
data_MILD = data_1[:,0][idx_MILD]

#   moderate dementia (MOD)
idx_MOD = np.where(np.logical_and(data_1a[:,1]>=MOD_MMSE[0],data_1a[:,1]<=MOD_MMSE[1]))
data_MOD = data_1[:,0][idx_MOD]

#   severe dementia (SEVD)
idx_SEVD = np.where(data_1a[:,1]<=SEV_MMSE)
data_SEVD = data_1[:,0][idx_SEVD]

"""
find the scans of the classes of dementia within the timepoints scans
"""

#   scans of no dementia across timepoints
def test(data_class, dataScans_TP,dataScans,scans):
    idx_class_TP = np.where(np.isin(([x[:-8] for x in dataScans_TP]), data_class))[0]
    init_sname = dataScans_TP[idx_class_TP]
    idx_scansClass_TP = np.where(np.isin(dataScans, init_sname))[0]
    scansClass_TP = scans[idx_scansClass_TP]  
    return scansClass_TP 

# no dementia (ND) scans at baseline
scansND_B = []
for i in range(len(data_ND)):
    scansND_B.append(test(data_ND[i], dataScans_B,dataScans,scans))
scans_runs_ND_B = list(chain.from_iterable(scansND_B))
init_scansND_B = [x[:-16] for x in scans_runs_ND_B]
idx_run2_scansND_B = [idx for idx, item in enumerate(init_scansND_B) if item in init_scansND_B[:idx]]
idx_run1_scansND_B = [idx for idx, item in enumerate(init_scansND_B) if item not in init_scansND_B[:idx]]
run1_scansND_B  = [scans_runs_ND_B[i] for i in idx_run1_scansND_B]
run2_scansND_B  = [scans_runs_ND_B[i] for i in idx_run2_scansND_B]



# no dementia (ND) scans at 4 months
scansND_4 = []
for i in range(len(data_ND)):
    scansND_4.append(test(data_ND[i], dataScans_4,dataScans,scans))
scansND_4 = np.array(scansND_4,dtype=object)
scans_runs_ND_4 = list(chain.from_iterable(scansND_4))
init_scansND_4 = [x[:-16] for x in scans_runs_ND_4]
idx_run2_scansND_4 = [idx for idx, item in enumerate(init_scansND_4) if item in init_scansND_4[:idx]]
idx_run1_scansND_4 = [idx for idx, item in enumerate(init_scansND_4) if item not in init_scansND_4[:idx]]
run1_scansND_4  = [scans_runs_ND_4[i] for i in idx_run1_scansND_4]
run2_scansND_4  = [scans_runs_ND_4[i] for i in idx_run2_scansND_4]


# no dementia (ND) scans at 8 months
scansND_8 = []
for i in range(len(data_ND)):
    scansND_8.append(test(data_ND[i], dataScans_8,dataScans,scans))
scansND_8 = np.array(scansND_8,dtype=object)
scans_runs_ND_8 = list(chain.from_iterable(scansND_8))
init_scansND_8 = [x[:-16] for x in scans_runs_ND_8]
idx_run2_scansND_8 = [idx for idx, item in enumerate(init_scansND_8) if item in init_scansND_8[:idx]]
idx_run1_scansND_8 = [idx for idx, item in enumerate(init_scansND_8) if item not in init_scansND_8[:idx]]
run1_scansND_8  = [scans_runs_ND_8[i] for i in idx_run1_scansND_8]
run2_scansND_8  = [scans_runs_ND_8[i] for i in idx_run2_scansND_8]


#  scans of mild dementia across timepoints

# mild dementia (MILD) scans at baseline
scansMILD_B = []
for i in range(len(data_MILD)):
    scansMILD_B.append(test(data_MILD[i], dataScans_B,dataScans,scans))
scansMILD_B = np.array(scansMILD_B,dtype=object)
scans_runs_MILD_B = list(chain.from_iterable(scansMILD_B))
init_scansMILD_B = [x[:-16] for x in scans_runs_MILD_B]
idx_run2_scansMILD_B = [idx for idx, item in enumerate(init_scansMILD_B) if item in init_scansMILD_B[:idx]]
idx_run1_scansMILD_B = [idx for idx, item in enumerate(init_scansMILD_B) if item not in init_scansMILD_B[:idx]]
run1_scansMILD_B  = [scans_runs_MILD_B[i] for i in idx_run1_scansMILD_B]
run2_scansMILD_B  = [scans_runs_MILD_B[i] for i in idx_run2_scansMILD_B]

# mild dementia (MILD) scans at 4 months
scansMILD_4 = []
for i in range(len(data_MILD)):
    scansMILD_4.append(test(data_MILD[i], dataScans_4,dataScans,scans))
scansMILD_4 = np.array(scansMILD_4,dtype=object)
scans_runs_MILD_4 = list(chain.from_iterable(scansMILD_4))
init_scansMILD_4 = [x[:-16] for x in scans_runs_MILD_4]
idx_run2_scansMILD_4 = [idx for idx, item in enumerate(init_scansMILD_4) if item in init_scansMILD_4[:idx]]
idx_run1_scansMILD_4 = [idx for idx, item in enumerate(init_scansMILD_4) if item not in init_scansMILD_4[:idx]]
run1_scansMILD_4  = [scans_runs_MILD_4[i] for i in idx_run1_scansMILD_4]
run2_scansMILD_4  = [scans_runs_MILD_4[i] for i in idx_run2_scansMILD_4]


# mild dementia (MILD) scans at 8 months
scansMILD_8 = []
for i in range(len(data_MILD)):
    scansMILD_8.append(test(data_MILD[i], dataScans_8,dataScans,scans))
scansMILD_8 = np.array(scansMILD_8,dtype=object)
scans_runs_MILD_8 = list(chain.from_iterable(scansMILD_8))
init_scansMILD_8 = [x[:-16] for x in scans_runs_MILD_8]
idx_run2_scansMILD_8 = [idx for idx, item in enumerate(init_scansMILD_8) if item in init_scansMILD_8[:idx]]
idx_run1_scansMILD_8 = [idx for idx, item in enumerate(init_scansMILD_8) if item not in init_scansMILD_8[:idx]]
run1_scansMILD_8  = [scans_runs_MILD_8[i] for i in idx_run1_scansMILD_8]
run2_scansMILD_8  = [scans_runs_MILD_8[i] for i in idx_run2_scansMILD_8]


#  scans of moderate dementia across timepoints

# moderate dementia (MOD) scans at baseline
scansMOD_B = []
for i in range(len(data_MOD)):
    scansMOD_B.append(test(data_MOD[i], dataScans_B,dataScans,scans))
scansMOD_B = np.array(scansMOD_B,dtype=object)
scans_runs_MOD_B = list(chain.from_iterable(scansMOD_B))
init_scansMOD_B = [x[:-16] for x in scans_runs_MOD_B]
idx_run2_scansMOD_B = [idx for idx, item in enumerate(init_scansMOD_B) if item in init_scansMOD_B[:idx]]
idx_run1_scansMOD_B = [idx for idx, item in enumerate(init_scansMOD_B) if item not in init_scansMOD_B[:idx]]
run1_scansMOD_B  = [scans_runs_MOD_B[i] for i in idx_run1_scansMOD_B]
run2_scansMOD_B  = [scans_runs_MOD_B[i] for i in idx_run2_scansMOD_B]


# moderate dementia (MOD) scans at 4 months
scansMOD_4 = []
for i in range(len(data_MOD)):
    scansMOD_4.append(test(data_MOD[i], dataScans_4,dataScans,scans))
scansMOD_4 = np.array(scansMOD_4,dtype=object)
scans_runs_MOD_4 = list(chain.from_iterable(scansMOD_4))
init_scansMOD_4 = [x[:-16] for x in scans_runs_MOD_4]
idx_run2_scansMOD_4 = [idx for idx, item in enumerate(init_scansMOD_4) if item in init_scansMOD_4[:idx]]
idx_run1_scansMOD_4 = [idx for idx, item in enumerate(init_scansMOD_4) if item not in init_scansMOD_4[:idx]]
run1_scansMOD_4  = [scans_runs_MOD_4[i] for i in idx_run1_scansMOD_4]
run2_scansMOD_4  = [scans_runs_MOD_4[i] for i in idx_run2_scansMOD_4]


# moderate dementia (MOD) scans at 8 months
scansMOD_8 = []
for i in range(len(data_MOD)):
    scansMOD_8.append(test(data_MOD[i], dataScans_8,dataScans,scans))
scansMOD_8 = np.array(scansMOD_8,dtype=object)
scans_runs_MOD_8 = list(chain.from_iterable(scansMOD_8))
init_scansMOD_8 = [x[:-16] for x in scans_runs_MOD_8]
idx_run2_scansMOD_8 = [idx for idx, item in enumerate(init_scansMOD_8) if item in init_scansMOD_8[:idx]]
idx_run1_scansMOD_8 = [idx for idx, item in enumerate(init_scansMOD_8) if item not in init_scansMOD_8[:idx]]
run1_scansMOD_8  = [scans_runs_MOD_8[i] for i in idx_run1_scansMOD_8]
run2_scansMOD_8  = [scans_runs_MOD_8[i] for i in idx_run2_scansMOD_8]


#  scans of severe dementia across timepoints

# severe dementia (SEVD) scans at baseline
scansSEVD_B = []
for i in range(len(data_SEVD)):
    scansSEVD_B.append(test(data_SEVD[i], dataScans_B,dataScans,scans))
scansSEVD_B = np.array(scansSEVD_B,dtype=object)
scans_runs_SEV_B = list(chain.from_iterable(scansSEVD_B))
init_scansSEVD_B = [x[:-16] for x in scans_runs_SEV_B]
idx_run2_scansSEVD_B = [idx for idx, item in enumerate(init_scansSEVD_B) if item in init_scansSEVD_B[:idx]]
idx_run1_scansSEVD_B = [idx for idx, item in enumerate(init_scansSEVD_B) if item not in init_scansSEVD_B[:idx]]
run1_scansSEVD_B  = [scans_runs_SEV_B[i] for i in idx_run1_scansSEVD_B]
run2_scansSEVD_B  = [scans_runs_SEV_B[i] for i in idx_run2_scansSEVD_B]


# severe dementia (SEVD) scans at 4 months
scansSEVD_4 = []
for i in range(len(data_SEVD)):
    scansSEVD_4.append(test(data_SEVD[i], dataScans_4,dataScans,scans))
scansSEVD_4 = np.array(scansSEVD_4,dtype=object)
scans_runs_SEV_4 = list(chain.from_iterable(scansSEVD_4))
init_scansSEVD_4 = [x[:-16] for x in scans_runs_SEV_4]
idx_run2_scansSEVD_4 = [idx for idx, item in enumerate(init_scansSEVD_4) if item in init_scansSEVD_4[:idx]]
idx_run1_scansSEVD_4 = [idx for idx, item in enumerate(init_scansSEVD_4) if item not in init_scansSEVD_4[:idx]]
run1_scansSEVD_4  = [scans_runs_SEV_4[i] for i in idx_run1_scansSEVD_4]
run2_scansSEVD_4  = [scans_runs_SEV_4[i] for i in idx_run2_scansSEVD_4]


# severe dementia (SEVD) scans at 8 months
scansSEVD_8 = []
for i in range(len(data_SEVD)):
    scansSEVD_8.append(test(data_SEVD[i], dataScans_8,dataScans,scans))
scansSEVD_8 = np.array(scansSEVD_8,dtype=object)
scans_runs_SEV_8 = list(chain.from_iterable(scansSEVD_8))
init_scansSEVD_8 = [x[:-16] for x in scans_runs_SEV_8]
idx_run2_scansSEVD_8 = [idx for idx, item in enumerate(init_scansSEVD_8) if item in init_scansSEVD_8[:idx]]
idx_run1_scansSEVD_8 = [idx for idx, item in enumerate(init_scansSEVD_8) if item not in init_scansSEVD_8[:idx]]
run1_scansSEVD_8  = [scans_runs_SEV_8[i] for i in idx_run1_scansSEVD_8]
run2_scansSEVD_8  = [scans_runs_SEV_8[i] for i in idx_run2_scansSEVD_8]










