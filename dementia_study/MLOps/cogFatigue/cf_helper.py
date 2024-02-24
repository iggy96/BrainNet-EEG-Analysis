from cf_libs import*
sys.path.insert(1, '/Users/joshuaighalo/Documents/GitHub/eegDementia')
from eeg_helper import*




def convertEpochsToPSD(epochs,fs,epochs_2D=False,epochs_3D=False):
    """
    Inputs:  (no of files,no of epochs,epoch length)
    Outputs:    1. (no of files,length of fft)
                2. frequencies
    """
    if epochs_2D:
        Epochs = np.nan_to_num(epochs)
        # transpose to get channels x time x trials
        freqs,EpochsPSD = psd(Epochs.T,fs,data_2D=True)
        freqs = freqs[0,:]
        # convert all zeros to nans
        EpochsPSD[np.where(EpochsPSD==0)] = np.nan
        # average across trials
        meanPSDs = np.nanmean(EpochsPSD,axis=0)
    if epochs_3D:
        # convert all nans to 0
        Epochs = np.nan_to_num(epochs)
        # transpose to get channels x time x trials
        EpochsNew = np.transpose(Epochs,(0,2,1))
        freqs,EpochsPSD = psd(EpochsNew,fs,data_3D=True)
        freqs = freqs[0,0,:]
        # convert all zeros to nans
        EpochsPSD[np.where(EpochsPSD==0)] = np.nan
        # average across trials
        meanPSDs = np.nanmean(EpochsPSD,axis=1)
    return meanPSDs,freqs,EpochsPSD

def slidingWindow(array,timing,window_size,step):
    #   Inputs  :   array    - 2D numpy array (d0 = samples, d1 = channels) of filtered EEG data
    #               window_size - size of window to be used for sliding
    #               freq   - step size for sliding window 
    #   Output  :   3D array (columns of array,no of windows,window size)
    def rolling_window(data_array,timing_array,window_size,step_size):
        idx_winSize = np.where(timing_array == window_size)[0][0]
        idx_stepSize = np.where(timing_array == step_size)[0][0]
        shape = (data_array.shape[0] - idx_winSize + 1, idx_winSize)
        strides = (data_array.strides[0],) + data_array.strides
        rolled = np.lib.stride_tricks.as_strided(data_array, shape=shape, strides=strides)
        return rolled[np.arange(0,shape[0],idx_stepSize)]
    out_final = []
    for i in range(len(array.T)):
        out_final.append(rolling_window(array[:,i],timing,window_size,step))
    out_final = np.asarray(out_final).T
    out_final = out_final.transpose()
    return out_final

def removeBadRawEEGs(filenames,version,localPath):
    # used to remove raw eegs with large artifacts
    def params(filename,version,localPath):
        device = importFile.neurocatch()
        fileObjects = device.init(version,filename,localPath,dispIMG=False)
        rawEEG = fileObjects[0]
        rawEOG = fileObjects[1]
        filtering = filters()
        adaptiveFilterOutput = filtering.adaptive(rawEEG,rawEOG)
        # compute mse between rawEEG and adaptiveFilterOutput
        mse = np.nanmean((rawEEG - adaptiveFilterOutput)**2)
        return mse
    mseScores = []
    for filename in filenames:
        mseScores.append(params(filename,version,localPath))
    mseScores = np.array(mseScores)
    # place filenames next to mseScores
    names_scores = np.vstack((mseScores,filenames))
    # print the names_scores in dataframe format
    df = pd.DataFrame(names_scores.T,columns=['mse','filename'])
    display(df)
    # check for outliers among the mse scores
    mseScores = names_scores[0]
    # get the mean of the mse scores
    mean = np.nanmean(mseScores.astype(float))
    std = np.nanstd(mseScores.astype(float))
    # get the z score of the mse scores
    zScores = (mseScores.astype(float) - mean)/std
    # get the index of the z scores that are greater than 3
    outliers = np.where(zScores > 3)
    outliers = outliers[0]
    # get the filenames of the outliers
    outlierNames = names_scores[1][outliers]
    # print the outlier names in dataframe format
    df = pd.DataFrame(outlierNames,columns=['outlier filenames'])
    display(df)
    # remove the outliers from the filenames
    for outlier in outlierNames:
        filenames.remove(outlier)
    return filenames  

def duplicateBetweenLists(list_1,list_2):
    # keep the first four characters of each items in the list   
    char_list_1 = [x[:4] for x in list_1]
    char_list_2 = [x[:4] for x in list_2]
    # keep duplicates between the two lists and their indices
    dup_list = [x for x in char_list_1 if x in char_list_2]
    # use char of duplicates to find original items for both lists
    dup_list_1 = [x for x in list_1 if x[:4] in dup_list]
    dup_list_2 = [x for x in list_2 if x[:4] in dup_list]
    # vertical stack the two lists
    dup_list = np.vstack((dup_list_1,dup_list_2))
    # display in dataframe
    df = pd.DataFrame({'Run 1':dup_list_1,'Run 2':dup_list_2})
    print(df)
    print("\n")
    return dup_list.T

def generatePSDsUsingEpochs(filenames_1,filenames_2,deviceVersion,path,sfreq,line,highPass,lowPass,stimTriggers,clip,channel_names,Epochs_MultipleSubjects):
    args = {'deviceVersion':deviceVersion,'path':path,'sfreq':sfreq,'line':line,'highPass':highPass,'lowPass':lowPass,'stimTriggers':stimTriggers,
            'clip':clip,'channel_names':channel_names,'Epochs_MultipleSubjects':Epochs_MultipleSubjects}

    mEpochsBND_1 = pipeline(filenames_1,**args)
    mEpochsTones_1 = mEpochsBND_1[0]
    mEpochsWords_1 = mEpochsBND_1[1]
    mEpochsFzStd_1 = mEpochsTones_1[:,0,:,:]
    mEpochsFzDev_1 = mEpochsTones_1[:,1,:,:]
    mEpochsCzStd_1 = mEpochsTones_1[:,2,:,:]
    mEpochsCzDev_1 = mEpochsTones_1[:,3,:,:]
    mEpochsPzStd_1 = mEpochsTones_1[:,4,:,:]
    mEpochsPzDev_1 = mEpochsTones_1[:,5,:,:]
    mEpochsFzCon_1 = mEpochsWords_1[:,0,:,:]
    mEpochsFzInc_1 = mEpochsWords_1[:,1,:,:]
    mEpochsCzCon_1 = mEpochsWords_1[:,2,:,:]
    mEpochsCzInc_1 = mEpochsWords_1[:,3,:,:]
    mEpochsPzCon_1 = mEpochsWords_1[:,4,:,:]
    mEpochsPzInc_1 = mEpochsWords_1[:,5,:,:]

    #       Run 2
    mEpochsBND_2 = pipeline(filenames_2,**args)
    mEpochsTones_2 = mEpochsBND_2[0]
    mEpochsWords_2 = mEpochsBND_2[1]
    mEpochsFzStd_2 = mEpochsTones_2[:,0,:,:]
    mEpochsFzDev_2 = mEpochsTones_2[:,1,:,:]
    mEpochsCzStd_2 = mEpochsTones_2[:,2,:,:]
    mEpochsCzDev_2 = mEpochsTones_2[:,3,:,:]
    mEpochsPzStd_2 = mEpochsTones_2[:,4,:,:]
    mEpochsPzDev_2 = mEpochsTones_2[:,5,:,:]
    mEpochsFzCon_2 = mEpochsWords_2[:,0,:,:]
    mEpochsFzInc_2 = mEpochsWords_2[:,1,:,:]
    mEpochsCzCon_2 = mEpochsWords_2[:,2,:,:]
    mEpochsCzInc_2 = mEpochsWords_2[:,3,:,:]
    mEpochsPzCon_2 = mEpochsWords_2[:,4,:,:]
    mEpochsPzInc_2 = mEpochsWords_2[:,5,:,:]


    #       Stack Epochs
    epochsFz_1 = np.concatenate((mEpochsFzStd_1,mEpochsFzDev_1,mEpochsFzCon_1,mEpochsFzInc_1),axis=1)
    epochsCz_1 = np.concatenate((mEpochsCzStd_1,mEpochsCzDev_1,mEpochsCzCon_1,mEpochsCzInc_1),axis=1)
    epochsPz_1 = np.concatenate((mEpochsPzStd_1,mEpochsPzDev_1,mEpochsPzCon_1,mEpochsPzInc_1),axis=1)
    epochsFz_2 = np.concatenate((mEpochsFzStd_2,mEpochsFzDev_2,mEpochsFzCon_2,mEpochsFzInc_2),axis=1)
    epochsCz_2 = np.concatenate((mEpochsCzStd_2,mEpochsCzDev_2,mEpochsCzCon_2,mEpochsCzInc_2),axis=1)
    epochsPz_2 = np.concatenate((mEpochsPzStd_2,mEpochsPzDev_2,mEpochsPzCon_2,mEpochsPzInc_2),axis=1)
    epochsRun_1 = np.stack((epochsFz_1,epochsCz_1,epochsPz_1),axis=0)
    epochsRun_2 = np.stack((epochsFz_2,epochsCz_2,epochsPz_2),axis=0)
    epochsGlobal_1 = np.mean(epochsRun_1,axis=0)
    epochsGlobal_2 = np.mean(epochsRun_2,axis=0)

    #       Convert epochs to PSDs
    psdFz_1,freqs,e = convertEpochsToPSD(epochsFz_1,sfreq,epochs_3D=True)
    psdCz_1,freqs,e = convertEpochsToPSD(epochsCz_1,sfreq,epochs_3D=True)
    psdPz_1,freqs,e = convertEpochsToPSD(epochsPz_1,sfreq,epochs_3D=True)
    psdFz_2,freqs,e = convertEpochsToPSD(epochsFz_2,sfreq,epochs_3D=True)
    psdCz_2,freqs,e = convertEpochsToPSD(epochsCz_2,sfreq,epochs_3D=True)
    psdPz_2,freqs,e = convertEpochsToPSD(epochsPz_2,sfreq,epochs_3D=True)
    psdGlobal_1,freqs,e = convertEpochsToPSD(epochsGlobal_1,sfreq,epochs_3D=True)
    psdGlobal_2,freqs,e = convertEpochsToPSD(epochsGlobal_2,sfreq,epochs_3D=True)
    
    freqs_ = np.tile(freqs,(len(psdFz_1),1)) 
    psd_run_1 = np.stack((psdFz_1,psdCz_1,psdPz_1,psdGlobal_1,freqs_),axis=0)
    freqs_ = np.tile(freqs,(len(psdFz_2),1))
    psd_run_2 = np.stack((psdFz_2,psdCz_2,psdPz_2,psdGlobal_2,freqs_),axis=0)
    return psd_run_1,psd_run_2

def mlpipeline(clf_name,clf,traindata,testdatasets,testdatasets_names):
    def params(model_name,model,trainset,testset):
        #   prepare trainset
        features = trainset.columns.values.tolist()
        features = trainset.columns.values.tolist()
        #trainset.drop(['index'], axis=1, inplace=True)
        trainset.fillna(trainset.mean(), inplace=True)
        #   prepare testset
        testset = testset.dropna()
        features = testset.columns.values.tolist()
        #testset.drop(['index'], axis=1, inplace=True)
        #   seperate features and target
        xtestset,ytestset = testset.drop(['labels'], axis=1), testset['labels'].ravel()
        xtrainset = trainset.drop(['labels'], axis=1)
        ytrainset = trainset['labels'].ravel()
        #   normalize
        scaler = MinMaxScaler()
        xtrainset_norm = scaler.fit_transform(xtrainset)
        xtestset_norm = scaler.transform(xtestset)
        xtrainset_norm = pd.DataFrame(xtrainset_norm)
        #xtrainset_norm.columns = features[1:-1]
        #   pca
        xtrainset_norm = xtrainset_norm.values
        pca = PCA(n_components=0.99,random_state=0)
        pca.fit(xtrainset_norm)
        xtrainset_pca = pca.transform(xtrainset_norm)
        xtestset_pca = pca.transform(xtestset_norm)
        #  fit model
        classifier=model
        model_norm,model_pca = classifier,classifier
        model_norm.fit(xtrainset_norm,ytrainset)
        ypred_norm = model_norm.predict(xtestset_norm)
        model_pca.fit(xtrainset_pca,ytrainset)
        ypred_pca = model_pca.predict(xtestset_pca)
        #   evaluate
        acc_norm = accuracy_score(ytestset, ypred_norm)
        tn, fp, fn, tp = confusion_matrix(ytestset, ypred_norm).ravel()
        spec_norm = tn/(tn+fp)
        sens_norm = tp/(tp+fn)
        acc_pca = accuracy_score(ytestset, ypred_pca)
        tn, fp, fn, tp = confusion_matrix(ytestset, ypred_pca).ravel()
        spec_pca = tn/(tn+fp)
        sens_pca = tp/(tp+fn)
        perf_norm = pd.DataFrame ({"Norm-Accuracy":acc_norm, "Norm-Specificity": spec_norm, "Norm-Sensitivity": sens_norm},index=[model_name])
        perf_pca = pd.DataFrame ({"PCA-Accuracy":acc_pca, "PCA-Specificity": spec_pca, "PCA-Sensitivity": sens_pca},index=[model_name])
        display(perf_norm)
        display(perf_pca)
        return perf_norm,perf_pca
    
    traindata.drop(['index'], axis=1, inplace=True)
    for i in range(len(testdatasets)):
        testdatasets[i].drop(['index'], axis=1, inplace=True)
    output = []
    for i in range(len(testdatasets)):
        print('\n',testdatasets_names[i])
        output.append(params(clf_name,clf,traindata,testdatasets[i]))
    return output
    headers = headers[1:-1]
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - (iqr * 1.5)
    upper_bound = q3 + (iqr * 1.5)
    data = np.where((data > upper_bound) | (data < lower_bound), np.nan, data)
    # fill missing values with median
    df = pd.DataFrame(data)
    df.fillna(df.median(), inplace=True)
    # convert numpy array to pandas dataframe
    df = pd.DataFrame(df)
    # add column names
    df.columns = headers
    # remove columns with all nan values
    print(df.columns[df.isna().all()].tolist())
    df = df.dropna(axis=1, how='all')
    return df