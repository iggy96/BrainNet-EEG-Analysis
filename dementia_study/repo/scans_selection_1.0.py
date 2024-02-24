"""
Input:  scans (variables) classfified by dementia class and timepoint from dementia_classes.py
Output: scans with the best eeg quality to represent each participants within a class

Note: -     each participant within the dementia classes have mutiple runs taken per timepoint 
            i.e., some participants have eeg taken from them twice per timepoint while it varies for others
      -     code is designed based on two runs and not three timepoints
      the low cut and highcut values of the signal quality function is fixed while the threshold is the parameter to be varied
"""
from fn_cfg import *
import params as cfg
import dementia_classes




print("NOTES:")
print("Pre-Quality Processing: Participants of each group have multiple scans per timepoints")
print("Post-Quality Processing: Participants of each group are allocated just one scan")
print("\n")


def scanSelection(device_version,run_IDs_1,run_IDs_2,threshold,channel_name,local_path,fs,line,dem_class,timepoint,dispSubStats,dispMainStats):

    device_version,run_IDs_1,run_IDs_2,threshold,channel_name,local_path,fs,line,dem_class,timepoint,dispSubStats,dispMainStats = device_version,run_IDs_1,run_IDs_2,threshold,channel_name,local_path,fs,line,dem_class,timepoint,dispSubStats,dispMainStats
    
    
    def eegSignalQuality(device_version,scan_ID,threshold,channel_name,local_path,fs,line,dispSubStats):
        """
        """
        chans = dict(Fz=0,Cz=1,Pz=2)
        version,filename,localPath = device_version,scan_ID,local_path 
        device = importFile.neurocatch()
        fileObjects = device.init(version,filename,localPath,dispIMG=False)
        rawEEG = fileObjects[0]
        rawEOG = fileObjects[1]
        filtering = filters()
        adaptiveFilterOutput = filtering.adaptive(rawEEG,rawEOG)
        notchFilterOutput = filtering.notch(adaptiveFilterOutput,line,fs)
        bandPassFilterOutput = filtering.butterBandPass(notchFilterOutput,lowcut=0.1,highcut=10,fs=fs)
        chan_idx = chans[channel_name]
        pkScore = quality_p2p(bandPassFilterOutput)
        pkScore = pkScore[chan_idx]
        if pkScore <= threshold:
            if dispSubStats == True:
                print(scan_ID,"quality: good")
            else:
                pass
            score = pkScore
            scan = scan_ID
        elif pkScore > threshold:
            if dispSubStats == True:
                print(scan_ID,"quality: bad")
            else:
                pass
            score = pkScore
            scan = 'rejected'
        if math.isnan(pkScore)==True:
            if dispSubStats == True:
                print(scan_ID,"quality: bad")
            else:
                pass
            score = pkScore
            scan = 'rejected'
        return scan,score


    print("Processing for",dem_class,'at',timepoint,'.............................................................')
    char_run_IDs_1 = [x[:-16] for x in run_IDs_1]
    char_run_IDs_2 = [x[:-16] for x in run_IDs_2]
    char_runs = char_run_IDs_1 + char_run_IDs_2
    no_dup_chars = list(dict.fromkeys(char_runs))

    if dispMainStats == True:
        print("Number of",dem_class,"participants pre-quality assessment:",len(no_dup_chars))
    else:
        pass

    if dispSubStats==True:
        print("Number of scans in run 1: ",len(run_IDs_1))
        print("Number of scans in run 2: ",len(run_IDs_2))
    else:
        pass

    status_runs_1 = []
    status_runs_2 = []
    scores_runs_1 = []
    scores_runs_2 = []
    for i in range(len(run_IDs_1)):
        scan_ID = run_IDs_1[i]
        scan,score = eegSignalQuality(version,scan_ID,threshold,channel,localPath,fs,line,dispSubStats)
        status_runs_1.append(scan)
        scores_runs_1.append(score)

    for i in range(len(run_IDs_2)):
        scan_ID = run_IDs_2[i]
        scan,score = eegSignalQuality(version,scan_ID,threshold,channel,localPath,fs,line,dispSubStats)
        status_runs_2.append(scan)
        scores_runs_2.append(score)

    idx_rejected_1 = [i for i,val in enumerate(status_runs_1) if val=='rejected']
    idx_accepted_1 = [i for i,val in enumerate(status_runs_1) if val!='rejected']
    idx_rejected_2 = [i for i,val in enumerate(status_runs_2) if val=='rejected']
    idx_accepted_2 = [i for i,val in enumerate(status_runs_2) if val!='rejected']
    runIDs_1_Rej = [run_IDs_1[i] for i in idx_rejected_1]
    runIDs_1_Acc = [run_IDs_1[i] for i in idx_accepted_1]
    runIDs_2_Rej = [run_IDs_2[i] for i in idx_rejected_2]
    runIDs_2_Acc = [run_IDs_2[i] for i in idx_accepted_2]
    scores_1_Acc = [scores_runs_1[i] for i in idx_accepted_1]
    scores_1_Rej = [scores_runs_1[i] for i in idx_rejected_1]
    scores_2_Acc = [scores_runs_2[i] for i in idx_accepted_2]
    scores_2_Rej = [scores_runs_2[i] for i in idx_rejected_2]

    if dispSubStats==True:
        print('\n')
        print("Number of accepted scans in run 1: ",len(runIDs_1_Acc))
        print("Number of rejected scans in run 1: ",len(runIDs_1_Rej))
        print('\n')
        print("Number of accepted scans in run 2: ",len(runIDs_2_Acc))
        print("Number of rejected scans in run 2: ",len(runIDs_2_Rej))
    else:
        pass

    merge_acc_runIDs = runIDs_1_Acc + runIDs_2_Acc
    merge_acc_scores = scores_1_Acc + scores_2_Acc

    char_merge_acc_runIDs = [x[:-16] for x in merge_acc_runIDs]

    none_duplicate_charIDs = [unique for unique in (list(set(char_merge_acc_runIDs))) if char_merge_acc_runIDs.count(unique) == 1]
    d = {item: idx for idx, item in enumerate(char_merge_acc_runIDs)}
    items_to_find = none_duplicate_charIDs
    idx_none_duplicate_charIDs =[d.get(item) for item in items_to_find]
    none_duplicate_runIDs = [merge_acc_runIDs[i] for i in idx_none_duplicate_charIDs]

    new_acc_runIDs = [v for i, v in enumerate(merge_acc_runIDs) if i not in idx_none_duplicate_charIDs]
    new_acc_scores = [v for i, v in enumerate(merge_acc_scores) if i not in idx_none_duplicate_charIDs]

    if len(new_acc_runIDs) > 1:
        rolling_scores = slidingWindow(np.array(new_acc_scores),int(len(new_acc_runIDs)/2),int(len(new_acc_runIDs)/2)).T
        rolling_runIDs = slidingWindow(np.array(new_acc_runIDs),int(len(new_acc_runIDs)/2),int(len(new_acc_runIDs)/2)).T
        idx_bestPeaks = np.argmin(rolling_scores,axis=1)
        bestPeaks_runIDs = [rolling_runIDs[i][j] for i,j in zip(range(len(rolling_runIDs)),idx_bestPeaks)]
        accepted_runIDs = bestPeaks_runIDs+none_duplicate_runIDs
        print("Number of",dem_class,"participants post-quality assessment:",len(accepted_runIDs))
        print('\n')
        return accepted_runIDs
    if len(new_acc_runIDs) <= 1:
        accepted_runIDs = new_acc_runIDs+none_duplicate_runIDs
        print("Number of",dem_class,"participants post-quality assessment:",len(accepted_runIDs))
        print("\n")
        return accepted_runIDs




#   baseline scans for different dementia classes
base_run1_ND = dementia_classes.run1_scansND_B
base_run2_ND = dementia_classes.run2_scansND_B
base_run1_MILD = dementia_classes.run1_scansMILD_B
base_run2_MILD = dementia_classes.run2_scansMILD_B
base_run1_MOD = dementia_classes.run1_scansMOD_B
base_run2_MOD = dementia_classes.run2_scansMOD_B
base_run1_SEVD = dementia_classes.run1_scansSEVD_B
base_run2_SEVD = dementia_classes.run2_scansSEVD_B

#   4-months scans for different dementia classes
four_run1_ND = dementia_classes.run1_scansND_4
four_run2_ND = dementia_classes.run2_scansND_4
four_run1_MILD = dementia_classes.run1_scansMILD_4
four_run2_MILD = dementia_classes.run2_scansMILD_4
four_run1_MOD = dementia_classes.run1_scansMOD_4
four_run2_MOD = dementia_classes.run2_scansMOD_4
four_run1_SEVD = dementia_classes.run1_scansSEVD_4
four_run2_SEVD = dementia_classes.run2_scansSEVD_4

#   8-months scans for different dementia classes
eight_run1_ND = dementia_classes.run1_scansND_8
eight_run2_ND = dementia_classes.run2_scansND_8
eight_run1_MILD = dementia_classes.run1_scansMILD_8
eight_run2_MILD = dementia_classes.run2_scansMILD_8
eight_run1_MOD = dementia_classes.run1_scansMOD_8
eight_run2_MOD = dementia_classes.run2_scansMOD_8
eight_run1_SEVD = dementia_classes.run1_scansSEVD_8
eight_run2_SEVD = dementia_classes.run2_scansSEVD_8


threshold = 300 #uV
localPath = '/Users/joshuaighalo/Downloads/EEG_Datasets/laurel_place/cleaned_dataset'
destinationPath = "/Users/joshuaighalo/Documents/BrainNet/Projects/Workspace/results/laurel place/quality scans"
destinationFileName = "300uV_Threshold@0.1_10Hz.csv"
version = 1.0
channel = 'Cz'
fs = cfg.fs
line = cfg.line
dispSubStats = False
dispMainStats = True

baseND = scanSelection(version,base_run1_ND,base_run2_ND,threshold,channel,localPath,fs,line,
                        'no dementia','baseline',dispSubStats,dispMainStats)
baseMILD = scanSelection(version,base_run1_MILD,base_run2_MILD,threshold,channel,localPath,fs,line,
                        'mild dementia','baseline',dispSubStats,dispMainStats)
baseMOD = scanSelection(version,base_run1_MOD,base_run2_MOD,threshold,channel,localPath,fs,line,
                        'moderate dementia','baseline',dispSubStats,dispMainStats)
baseSEVD = scanSelection(version,base_run1_SEVD,base_run2_SEVD,threshold,channel,localPath,fs,line,
                        'severe dementia','baseline',dispSubStats,dispMainStats)
fourND = scanSelection(version,four_run1_ND,four_run2_ND,threshold,channel,localPath,fs,line,
                        'no dementia','4-months',dispSubStats,dispMainStats)
fourMILD = scanSelection(version,four_run1_MILD,four_run2_MILD,threshold,channel,localPath,fs,line,
                        'mild dementia','4-months',dispSubStats,dispMainStats)
fourMOD = scanSelection(version,four_run1_MOD,four_run2_MOD,threshold,channel,localPath,fs,line,
                        'moderate dementia','4-months',dispSubStats,dispMainStats)
fourSEVD = scanSelection(version,four_run1_SEVD,four_run2_SEVD,threshold,channel,localPath,fs,line,
                        'severe dementia','4-months',dispSubStats,dispMainStats)
eightND = scanSelection(version,eight_run1_ND,eight_run2_ND,threshold,channel,localPath,fs,line,
                        'no dementia','8-months',dispSubStats,dispMainStats)
eightMILD = scanSelection(version,eight_run1_MILD,eight_run2_MILD,threshold,channel,localPath,fs,line,
                        'mild dementia','8-months',dispSubStats,dispMainStats)
eightMOD = scanSelection(version,eight_run1_MOD,eight_run2_MOD,threshold,channel,localPath,fs,line,
                        'moderate dementia','8-months',dispSubStats,dispMainStats)
eightSEVD = scanSelection(version,eight_run1_SEVD,eight_run2_SEVD,threshold,channel,localPath,fs,line,
                        'severe dementia','8-months',dispSubStats,dispMainStats)

# export to csv
df1 = pd.DataFrame({"Baseline No Dementia": baseND})
df2 = pd.DataFrame({"Baseline Mild Dementia": baseMILD})
df3 = pd.DataFrame({"Baseline Moderate Dementia": baseMOD})
df4 = pd.DataFrame({"Baseline Severe Dementia": baseSEVD})
df5 = pd.DataFrame({"4-months No Dementia": fourND})
df6 = pd.DataFrame({"4-months Mild Dementia": fourMILD})
df7 = pd.DataFrame({"4-months Moderate Dementia": fourMOD})
df8 = pd.DataFrame({"4-months Severe Dementia": fourSEVD})
df9 = pd.DataFrame({"8-months No Dementia": eightND})
df10 = pd.DataFrame({"8-months Mild Dementia": eightMILD})
df11 = pd.DataFrame({"8-months Moderate Dementia": eightMOD})
df12 = pd.DataFrame({"8-months Severe Dementia": eightSEVD})
df13 = pd.concat([df1,df2,df3,df4,df5,df6,df7,df8,df9,df10,df11,df12], ignore_index=False, axis=1)
df13.to_csv(destinationPath + '/' + destinationFileName, index=False)
print("Done")