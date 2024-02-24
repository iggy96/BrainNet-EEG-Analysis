"""
Joshua Ighalo 12/06/2022

dementia study scans sorter

this script sorts the scans into the following categories:
1. severe dementia
2. moderate dementia
3. mild dementia
4. no dementia

Input: .xlsx file with the following columns: scan ID and MMSE

This script is custom built for only the laurel place dataset and will not work for other datasets
It is an improved version of the scans_seperator_10.py & dementia_stages_10.py scripts with their functionalities merged together
       
"""
#%%
import sys
sys.path.insert(1, '/Users/joshuaighalo/Documents/GitHub/eegDementia')
from eeg_libs import *

def params(mmse_scale,null_scans,clinical_filename,scaninfo_filename,search_path):
    df_1 = pd.read_excel (search_path +clinical_filename)
    df_2 = pd.read_excel(search_path + scaninfo_filename)
    data_1 = (df_1[['ID','MMSE']]).to_numpy()
    data_2 = (df_2[['Baseline','4 months','8 months']]).to_numpy()

    #   remove scans of participants who died
    died = null_scans
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
    idx_ND = np.where(np.logical_and(data_1a[:,1]>=mmse_scale[0],data_1a[:,1]<=mmse_scale[1]))
    data_ND = data_1[:,0][idx_ND]

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

    #   baseline scans for different dementia classes
    base_NO,four_NO,eight_NO = scans_runs_ND_B,scans_runs_ND_4,scans_runs_ND_8


    #   STRIP CHAR: remove the last fourteen characters of scan names
    init_base_NO = [x[:-14] for x in base_NO]
    init_four_NO = [x[:-14] for x in four_NO]
    init_eight_NO = [x[:-14] for x in eight_NO]


    #   FIND INDICES: indices of init scans that end with "_2" representing second scans
    idxRun2_baseNO = [i for i, x in enumerate(init_base_NO) if x.endswith('_2')]
    idxRun2_fourNO = [i for i, x in enumerate(init_four_NO) if x.endswith('_2')]
    idxRun2_eightNO = [i for i, x in enumerate(init_eight_NO) if x.endswith('_2')]


    #   REMOVE RUNS 2: removes run 2 from the list of scans
    base_NO_1 = [x for i, x in enumerate(base_NO) if i not in idxRun2_baseNO]
    four_NO_1 = [x for i, x in enumerate(four_NO) if i not in idxRun2_fourNO]
    eight_NO_1 = [x for i, x in enumerate(eight_NO) if i not in idxRun2_eightNO]
    print("all first scans grouped!")


    #%% 
    #   baseline scans for different dementia classes
    base_NO,four_NO,eight_NO = scans_runs_ND_B,scans_runs_ND_4,scans_runs_ND_8
    
    #   STRIP CHAR: remove the last fourteen characters of scan names
    init_base_NO = [x[:-14] for x in base_NO]
    init_four_NO = [x[:-14] for x in four_NO]
    init_eight_NO = [x[:-14] for x in eight_NO]

    #   FIND INDICES: indices of init scans that end with "_2" representing second scans
    idxRun2_baseNO = [i for i, x in enumerate(init_base_NO) if x.endswith('_1')]
    idxRun2_fourNO = [i for i, x in enumerate(init_four_NO) if x.endswith('_1')]
    idxRun2_eightNO = [i for i, x in enumerate(init_eight_NO) if x.endswith('_1')]

    #   REMOVE RUNS 2: removes run 2 from the list of scans
    base_NO_2 = [x for i, x in enumerate(base_NO) if i not in idxRun2_baseNO]
    four_NO_2 = [x for i, x in enumerate(four_NO) if i not in idxRun2_fourNO]
    eight_NO_2 = [x for i, x in enumerate(eight_NO) if i not in idxRun2_eightNO]
    print("all second scans grouped!")
    return base_NO_1,four_NO_1,eight_NO_1,base_NO_2,four_NO_2,eight_NO_2







