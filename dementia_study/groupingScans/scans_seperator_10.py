"""
Laurel Place Study: select the first scans of all participants per timepoint

"""


import sys
sys.path.insert(1, '/Users/joshuaighalo/Documents/GitHub/eegDementia')
from eeg_libs import *
import dementia_stages_10 as dc

localPath = '/Users/joshuaighalo/Downloads/brainNet_datasets/laurel_place/cleaned_dataset'
destinationPath = "/Users/joshuaighalo/Documents/GitHub/eegDementia/groupingScans/database"
destinationFileName = "groupFirstScans.csv"

#   baseline scans for different dementia classes
base_NO = dc.scans_runs_ND_B
base_MILD = dc.scans_runs_MILD_B
base_MOD = dc.scans_runs_MOD_B
base_SEVERE = dc.scans_runs_SEV_B

#   4-months scans for different dementia classes
four_NO = dc.scans_runs_ND_4
four_MILD = dc.scans_runs_MILD_4
four_MOD = dc.scans_runs_MOD_4
four_SEVERE = dc.scans_runs_SEV_4

#  8-months scans for different dementia classes
eight_NO = dc.scans_runs_ND_8
eight_MILD = dc.scans_runs_MILD_8
eight_MOD = dc.scans_runs_MOD_8
eight_SEVERE = dc.scans_runs_SEV_8


#   STRIP CHAR: remove the last fourteen characters of scan names

#   Strip Char: NO
init_base_NO = [x[:-14] for x in base_NO]
init_four_NO = [x[:-14] for x in four_NO]
init_eight_NO = [x[:-14] for x in eight_NO]

#  Strip Char: MILD
init_base_MILD = [x[:-14] for x in base_MILD]
init_four_MILD = [x[:-14] for x in four_MILD]
init_eight_MILD = [x[:-14] for x in eight_MILD]

#  Strip Char: MOD
init_base_MOD = [x[:-14] for x in base_MOD]
init_four_MOD = [x[:-14] for x in four_MOD]
init_eight_MOD = [x[:-14] for x in eight_MOD]

#  Strip Char: SEVERE
init_base_SEVERE = [x[:-14] for x in base_SEVERE]
init_four_SEVERE = [x[:-14] for x in four_SEVERE]
init_eight_SEVERE = [x[:-14] for x in eight_SEVERE]


#   FIND INDICES: indices of init scans that end with "_2" representing second scans

#   Find Indices: NO
idxRun2_baseNO = [i for i, x in enumerate(init_base_NO) if x.endswith('_2')]
idxRun2_fourNO = [i for i, x in enumerate(init_four_NO) if x.endswith('_2')]
idxRun2_eightNO = [i for i, x in enumerate(init_eight_NO) if x.endswith('_2')]

#   Find Indices: MILD
idxRun2_baseMILD = [i for i, x in enumerate(init_base_MILD) if x.endswith('_2')]
idxRun2_fourMILD = [i for i, x in enumerate(init_four_MILD) if x.endswith('_2')]
idxRun2_eightMILD = [i for i, x in enumerate(init_eight_MILD) if x.endswith('_2')]

#   Find Indices: MOD
idxRun2_baseMOD = [i for i, x in enumerate(init_base_MOD) if x.endswith('_2')]
idxRun2_fourMOD = [i for i, x in enumerate(init_four_MOD) if x.endswith('_2')]
idxRun2_eightMOD = [i for i, x in enumerate(init_eight_MOD) if x.endswith('_2')]

#   Find Indices: SEVERE
idxRun2_baseSEVERE = [i for i, x in enumerate(init_base_SEVERE) if x.endswith('_2')]
idxRun2_fourSEVERE = [i for i, x in enumerate(init_four_SEVERE) if x.endswith('_2')]
idxRun2_eightSEVERE = [i for i, x in enumerate(init_eight_SEVERE) if x.endswith('_2')]


#   REMOVE RUNS 2: removes run 2 from the list of scans

#   Remove Run 2: NO
base_NO_1 = [x for i, x in enumerate(base_NO) if i not in idxRun2_baseNO]
four_NO_1 = [x for i, x in enumerate(four_NO) if i not in idxRun2_fourNO]
eight_NO_1 = [x for i, x in enumerate(eight_NO) if i not in idxRun2_eightNO]

#   Remove Run 2: MILD
base_MILD_1 = [x for i, x in enumerate(base_MILD) if i not in idxRun2_baseMILD]
four_MILD_1 = [x for i, x in enumerate(four_MILD) if i not in idxRun2_fourMILD]
eight_MILD_1 = [x for i, x in enumerate(eight_MILD) if i not in idxRun2_eightMILD]

#   Remove Run 2: MOD
base_MOD_1 = [x for i, x in enumerate(base_MOD) if i not in idxRun2_baseMOD]
four_MOD_1 = [x for i, x in enumerate(four_MOD) if i not in idxRun2_fourMOD]
eight_MOD_1 = [x for i, x in enumerate(eight_MOD) if i not in idxRun2_eightMOD]

#   Remove Run 2: SEVERE
base_SEVERE_1 = [x for i, x in enumerate(base_SEVERE) if i not in idxRun2_baseSEVERE]
four_SEVERE_1 = [x for i, x in enumerate(four_SEVERE) if i not in idxRun2_fourSEVERE]
eight_SEVERE_1 = [x for i, x in enumerate(eight_SEVERE) if i not in idxRun2_eightSEVERE]


#   EXPORT: export the list of scans to a csv file
df1 = pd.DataFrame({"Baseline No Dementia": base_NO})
df2 = pd.DataFrame({"Baseline Mild Dementia": base_MILD})
df3 = pd.DataFrame({"Baseline Moderate Dementia": base_MOD})
df4 = pd.DataFrame({"Baseline Severe Dementia": base_SEVERE})
df5 = pd.DataFrame({"4-months No Dementia": four_NO})
df6 = pd.DataFrame({"4-months Mild Dementia": four_MILD})
df7 = pd.DataFrame({"4-months Moderate Dementia": four_MOD})
df8 = pd.DataFrame({"4-months Severe Dementia": four_SEVERE})
df9 = pd.DataFrame({"8-months No Dementia": eight_NO})
df10 = pd.DataFrame({"8-months Mild Dementia": eight_MILD})
df11 = pd.DataFrame({"8-months Moderate Dementia": eight_MOD})
df12 = pd.DataFrame({"8-months Severe Dementia": eight_SEVERE})
df13 = pd.concat([df1,df2,df3,df4,df5,df6,df7,df8,df9,df10,df11,df12], ignore_index=False, axis=1)
df13.to_csv(destinationPath + '/' + destinationFileName, index=False)
print("all first scans grouped!")




#%% 


localPath = '/Users/joshuaighalo/Downloads/brainNet_datasets/laurel_place/cleaned_dataset'
destinationPath = "/Users/joshuaighalo/Documents/GitHub/eegDementia/groupingScans/database"
destinationFileName = "groupSecondScans.csv"

#   baseline scans for different dementia classes
base_NO = dc.scans_runs_ND_B
base_MILD = dc.scans_runs_MILD_B
base_MOD = dc.scans_runs_MOD_B
base_SEVERE = dc.scans_runs_SEV_B

#   4-months scans for different dementia classes
four_NO = dc.scans_runs_ND_4
four_MILD = dc.scans_runs_MILD_4
four_MOD = dc.scans_runs_MOD_4
four_SEVERE = dc.scans_runs_SEV_4

#  8-months scans for different dementia classes
eight_NO = dc.scans_runs_ND_8
eight_MILD = dc.scans_runs_MILD_8
eight_MOD = dc.scans_runs_MOD_8
eight_SEVERE = dc.scans_runs_SEV_8


#   STRIP CHAR: remove the last fourteen characters of scan names

#   Strip Char: NO
init_base_NO = [x[:-14] for x in base_NO]
init_four_NO = [x[:-14] for x in four_NO]
init_eight_NO = [x[:-14] for x in eight_NO]

#  Strip Char: MILD
init_base_MILD = [x[:-14] for x in base_MILD]
init_four_MILD = [x[:-14] for x in four_MILD]
init_eight_MILD = [x[:-14] for x in eight_MILD]

#  Strip Char: MOD
init_base_MOD = [x[:-14] for x in base_MOD]
init_four_MOD = [x[:-14] for x in four_MOD]
init_eight_MOD = [x[:-14] for x in eight_MOD]

#  Strip Char: SEVERE
init_base_SEVERE = [x[:-14] for x in base_SEVERE]
init_four_SEVERE = [x[:-14] for x in four_SEVERE]
init_eight_SEVERE = [x[:-14] for x in eight_SEVERE]


#   FIND INDICES: indices of init scans that end with "_2" representing second scans

#   Find Indices: NO
idxRun2_baseNO = [i for i, x in enumerate(init_base_NO) if x.endswith('_1')]
idxRun2_fourNO = [i for i, x in enumerate(init_four_NO) if x.endswith('_1')]
idxRun2_eightNO = [i for i, x in enumerate(init_eight_NO) if x.endswith('_1')]

#   Find Indices: MILD
idxRun2_baseMILD = [i for i, x in enumerate(init_base_MILD) if x.endswith('_1')]
idxRun2_fourMILD = [i for i, x in enumerate(init_four_MILD) if x.endswith('_1')]
idxRun2_eightMILD = [i for i, x in enumerate(init_eight_MILD) if x.endswith('_1')]

#   Find Indices: MOD
idxRun2_baseMOD = [i for i, x in enumerate(init_base_MOD) if x.endswith('_1')]
idxRun2_fourMOD = [i for i, x in enumerate(init_four_MOD) if x.endswith('_1')]
idxRun2_eightMOD = [i for i, x in enumerate(init_eight_MOD) if x.endswith('_1')]

#   Find Indices: SEVERE
idxRun2_baseSEVERE = [i for i, x in enumerate(init_base_SEVERE) if x.endswith('_1')]
idxRun2_fourSEVERE = [i for i, x in enumerate(init_four_SEVERE) if x.endswith('_1')]
idxRun2_eightSEVERE = [i for i, x in enumerate(init_eight_SEVERE) if x.endswith('_1')]


#   REMOVE RUNS 2: removes run 2 from the list of scans

#   Remove Run 2: NO
base_NO_2 = [x for i, x in enumerate(base_NO) if i not in idxRun2_baseNO]
four_NO_2 = [x for i, x in enumerate(four_NO) if i not in idxRun2_fourNO]
eight_NO_2 = [x for i, x in enumerate(eight_NO) if i not in idxRun2_eightNO]

#   Remove Run 2: MILD
base_MILD_2 = [x for i, x in enumerate(base_MILD) if i not in idxRun2_baseMILD]
four_MILD_2 = [x for i, x in enumerate(four_MILD) if i not in idxRun2_fourMILD]
eight_MILD_2 = [x for i, x in enumerate(eight_MILD) if i not in idxRun2_eightMILD]

#   Remove Run 2: MOD
base_MOD_2 = [x for i, x in enumerate(base_MOD) if i not in idxRun2_baseMOD]
four_MOD_2 = [x for i, x in enumerate(four_MOD) if i not in idxRun2_fourMOD]
eight_MOD_2 = [x for i, x in enumerate(eight_MOD) if i not in idxRun2_eightMOD]

#   Remove Run 2: SEVERE
base_SEVERE_2 = [x for i, x in enumerate(base_SEVERE) if i not in idxRun2_baseSEVERE]
four_SEVERE_2 = [x for i, x in enumerate(four_SEVERE) if i not in idxRun2_fourSEVERE]
eight_SEVERE_2 = [x for i, x in enumerate(eight_SEVERE) if i not in idxRun2_eightSEVERE]


#   EXPORT: export the list of scans to a csv file
df1 = pd.DataFrame({"Baseline No Dementia": base_NO})
df2 = pd.DataFrame({"Baseline Mild Dementia": base_MILD})
df3 = pd.DataFrame({"Baseline Moderate Dementia": base_MOD})
df4 = pd.DataFrame({"Baseline Severe Dementia": base_SEVERE})
df5 = pd.DataFrame({"4-months No Dementia": four_NO})
df6 = pd.DataFrame({"4-months Mild Dementia": four_MILD})
df7 = pd.DataFrame({"4-months Moderate Dementia": four_MOD})
df8 = pd.DataFrame({"4-months Severe Dementia": four_SEVERE})
df9 = pd.DataFrame({"8-months No Dementia": eight_NO})
df10 = pd.DataFrame({"8-months Mild Dementia": eight_MILD})
df11 = pd.DataFrame({"8-months Moderate Dementia": eight_MOD})
df12 = pd.DataFrame({"8-months Severe Dementia": eight_SEVERE})
df13 = pd.concat([df1,df2,df3,df4,df5,df6,df7,df8,df9,df10,df11,df12], ignore_index=False, axis=1)
df13.to_csv(destinationPath + '/' + destinationFileName, index=False)
print("all second scans grouped!")