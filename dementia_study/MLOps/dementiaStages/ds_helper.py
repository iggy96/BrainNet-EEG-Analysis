
from ds_libs import*
sys.path.insert(1, '/Users/joshuaighalo/Documents/GitHub/eegDementia')
from eeg_helper import*


def removeBadRawEEGs(filenames,version,fileInfo,localPath):
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
    # print the number of files before removing outliers
    print(fileInfo)
    print('Number of files before removing outliers: '+str(len(names_scores.T)))
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
    # print the number of files after removing outliers
    print('\n','Number of files after removing outliers: '+str(len(filenames)-len(outlierNames)))
    df = pd.DataFrame(outlierNames,columns=['outlier filenames'])
    display(df)
    # remove the outliers from the filenames
    for outlier in outlierNames:
        filenames.remove(outlier)
    print('\n')
    return filenames  

def psdAbsoluteBandPower(data_psd,data_freq,low,high):
    def params(psd,freqs,low,high): 
        freq_res = freqs[1] - freqs[0]
        idx_band = np.logical_and(freqs >= low, freqs <= high)
        bp = simps(psd[idx_band], dx=freq_res)  
        #bp /= simps(psd, dx=freq_res)
        return bp

    avg_BandPower = []
    for i in range(len(data_psd.T)):
        avg_BandPower.append(params(data_psd[:,i],data_freq,low,high))
    avg_BandPower= np.array(avg_BandPower).T
    avg_BandPower = np.nan_to_num(avg_BandPower, nan=np.nanmean(avg_BandPower))
    return avg_BandPower

def psdRelativeBandPower(data_psd,data_freq,low,high):
    def params(psd,freqs,low,high): 
        freq_res = freqs[1] - freqs[0]
        idx_band = np.logical_and(freqs >= low, freqs <= high)
        bp = simps(psd[idx_band], dx=freq_res)  
        bp /= simps(psd, dx=freq_res)
        return bp

    avg_BandPower = []
    for i in range(len(data_psd.T)):
        avg_BandPower.append(params(data_psd[:,i],data_freq,low,high))
    avg_BandPower= np.array(avg_BandPower).T
    avg_BandPower = np.nan_to_num(avg_BandPower, nan=np.nanmean(avg_BandPower))
    return avg_BandPower

def psdPeakPower(data_psd,data_freq,low,high):
    def params(psd,freqs,low,high):
        idxFreq_1 = np.abs(freqs - low).argmin()
        idxFreq_2 = np.abs(freqs - high).argmin()
        psdRange = psd[idxFreq_1:idxFreq_2]
        peakPower = np.max(psdRange)
        return peakPower  
    
    peakPower = []
    for i in range(len(data_psd.T)):
        peakPower.append(params(data_psd[:,i],data_freq,low,high))
    peakPower= np.array(peakPower).T
    peakPower = np.nan_to_num(peakPower, nan=np.nanmean(peakPower))
    return peakPower
     
def computePSD(data,fs,data_type):
    """
    Inputs: data - 1D, 2D or 3D numpy array
                    1D - single channel
                    2D - (samples,channels)
                    3D - (files,samples,channels)
            fs - sampling frequency
            data_1D - boolean, True if data is 1D
            data_2D - boolean, True if data is 2D
            data_3D - boolean, True if data is 3D
    Outputs: psd - 1D, 2D or 3D numpy array
    """
    def params_1D(dataIN,fs):
        psd, freqs = psd_array_multitaper(dataIN, fs,verbose=0)
        return freqs,psd
    def params_2D(dataIN,fs):
        freqs,psd = [],[]
        for i in range(len(dataIN.T)):
            freqs.append(params_1D(dataIN[:,i],fs)[0])
            psd.append(params_1D(dataIN[:,i],fs)[1])
        return np.array(freqs),np.array(psd)

    if data_type == '1D':
        frequency,powerspectraldensity = params_1D(data,fs)
    if data_type == '2D':
        frequency,powerspectraldensity = params_2D(data,fs)
    if data_type == '3D':
        frequency,powerspectraldensity = [],[]
        for i in range(len(data)):
            frequency.append(params_2D(data[i,:,:],fs)[0])
            powerspectraldensity.append(params_2D(data[i,:,:],fs)[1])
    return np.array(frequency),np.array(powerspectraldensity)

def extractEEG(filenames,version,localPath,freq,linefreq,highpass,lowpass):
    def params(device_version,scan_ID,local_path,fs,line_,lowcut,highcut):
        device = importFile.neurocatch()
        fileObjects = device.init(device_version,scan_ID,local_path,dispIMG=False)
        rawEEG = fileObjects[0]
        filtering = filters()
        notchFilterOutput = filtering.notch(rawEEG,line_,fs)
        bandPassFilterOutput = filtering.butterBandPass(notchFilterOutput,lowcut,highcut,fs)
        bandPassFilterOutput = bandPassFilterOutput[0:155000]
        return bandPassFilterOutput
    eeg = []
    for i in range(len(filenames)):
        eeg.append(params(version,filenames[i],localPath,freq,linefreq,highpass,lowpass))
    eeg = np.array(eeg)
    return eeg

def ar_maximumgradient(input_2D,threshold_value,timearray,len_window,step_size,choice_numwindows,channels):
    def params(data1D,threshold,time_array,winsize,step,numwindows,chan_title):
        def maxgrad2D(data2D):
            diff_succ_val = []
            for i in range(data2D.shape[0]):
                diff_succ_val.append(np.max(np.diff(data2D[i,:])))
            return np.array(diff_succ_val)

        def slidingwindow(data_1D,timing_array,window_size,step_size):
            idx_winsize = np.where(timing_array == window_size)[0][0]
            idx_stepsize = np.where(timing_array == step_size)[0][0]
            frame_len, hop_len = idx_winsize,idx_stepsize
            frames = librosa.util.frame(data_1D, frame_length=frame_len, hop_length=hop_len)
            windowed_frames = (np.hanning(frame_len).reshape(-1, 1)*frames).T
            return windowed_frames

        wins2D = slidingwindow(data1D,time_array,winsize,step)
        maxgrads = maxgrad2D(wins2D)
        highest_maxgrads = np.amax(maxgrads)
        lowest_maxgrads = np.amin(maxgrads)
        print('maximum gradient value of worst segment for %s is %f' % (chan_title,highest_maxgrads))
        print('minimum gradient value of best segment for %s is %f' % (chan_title,lowest_maxgrads))
        idxs_badMaxGrads = np.where(maxgrads > threshold)[0]
        idxs_cleanMaxGrads = np.where(maxgrads <= threshold)[0]
        idx_highBadMaxGrad = np.where(maxgrads == highest_maxgrads)[0][0]
        idx_highCleanMaxGrad = np.where(maxgrads == lowest_maxgrads)[0][0]
        bad_dataset = wins2D[idxs_badMaxGrads,:]
        clean_dataset = wins2D[idxs_cleanMaxGrads,:]
        remain_windows = len(wins2D) - len(clean_dataset)
        clean_dataset = np.concatenate((clean_dataset,np.full((remain_windows,wins2D.shape[1]),np.nan)),axis=0)
        print('total non-artifactual segments for %s is %d' % (chan_title,len(idxs_cleanMaxGrads)))
        print('total artifactual segments for %s is %d' % (chan_title,len(idxs_badMaxGrads)))
        clean_dataset = clean_dataset[0:numwindows,:]
        if clean_dataset.size == 0:
            print('no clean data for %s' % chan_title)
            clean_dataset = np.full((numwindows,wins2D.shape[1]),np.nan)
        print('total chosen non-artifactual segments for %s is %d' % (chan_title,len(clean_dataset)))
        worst_data, best_data = wins2D[idx_highBadMaxGrad,:], wins2D[idx_highCleanMaxGrad,:]
        fig,ax = plt.subplots(1,2,figsize=(10,3))
        fig.suptitle(chan_title)
        ax[0].plot(worst_data,color='red')
        ax[0].set_title('Worst data')
        ax[0].set_ylabel('Amplitude')
        ax[1].plot(best_data,color='green')
        ax[1].set_title('Best data')
        ax[1].set_ylabel('Amplitude')
        plt.show()
        return clean_dataset

    output_2D = []
    for i in range(input_2D.shape[1]):
        output_2D.append(params(input_2D[:,i],threshold_value,timearray,len_window,step_size,choice_numwindows,channels[i]))
    return np.array(output_2D)

def multipleArtfRemoval(input_3d,threshold_value,timearray,len_window,step_size,choice_numwindows,channels,group):
    print(group)
    artf_out = []
    for i in range(input_3d.shape[0]):
        artf_out.append(ar_maximumgradient(input_3d[i,:,:],threshold_value,timearray,len_window,step_size,choice_numwindows,channels))
    print('***************************************************************************************************************************************************************')
    return np.array(artf_out)
 
def multiplePSD(data_4D,fs):
    freqs,psd = [],[]
    for i in range(data_4D.shape[0]):
        freqs.append(computePSD(data_4D[i],fs,data_type='3D')[0])
        psd.append(computePSD(data_4D[i],fs,data_type='3D')[1])
    psd = np.array(psd)
    freqs = np.array(freqs)
    return freqs,psd

def multipleRelativeBandPower(input_3d,freq_array,low,high):
    avg_out = []
    for i in range(input_3d.shape[0]):
        avg_out.append(psdRelativeBandPower(input_3d[i,:,:],freq_array,low,high))
    return np.array(avg_out) 

def multipleAbsoluteBandPower(input_3d,freq_array,low,high):
    avg_out = []
    for i in range(input_3d.shape[0]):
        avg_out.append(psdAbsoluteBandPower(input_3d[i,:,:],freq_array,low,high))
    return np.array(avg_out) 

def multiplePeakPower(input_3d,freq_array,low,high):
    avg_out = []
    for i in range(input_3d.shape[0]):
        avg_out.append(psdPeakPower(input_3d[i,:,:],freq_array,low,high))
    return np.array(avg_out)

def detect_remove_outliers(df):
    fig, ax = plt.subplots(1, 1, figsize=(25,4))
    sns.boxplot(data=df, ax=ax)
    plt.xticks(rotation=90)
    plt.title("Before removing outliers")
    plt.show()
    sorted(df)
    q1, q3 = np.percentile(df, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - (iqr * 1.5)
    upper_bound = q3 + (iqr * 1.5)
    df[df < lower_bound] = np.nan
    df[df > upper_bound] = np.nan
    df.fillna(df.median(), inplace=True)
    # remove columns with all nan values
    df.dropna(axis=1, how='all', inplace=True)
    new_features = df.columns
    fig, ax = plt.subplots(1, 1, figsize=(25,4))
    sns.boxplot(data=df, ax=ax)
    plt.xticks(rotation=90)
    plt.title("After removing outliers")
    plt.show()
    return df, new_features
