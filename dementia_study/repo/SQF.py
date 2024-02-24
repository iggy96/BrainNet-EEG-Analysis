from eeg_libs import*

"""
signal quality frmaework is based on two papers:
1. Good data? The EEG Quality Index for Automated Assessment of Signal Quality
2. A semi-simulated EEG/EOG dataset for the comparison of EOG artifact rejection techniques
    https://www.sciencedirect.com/science/article/pii/S2352340916304000?via%3Dihub
    EEG data were obtained from twenty-seven healthy subjects, 14 males (mean age: 28.2 ± 7.5) and 
    13 females (mean age: 27.1±5.2), during an eyes-closed session. Nineteen EEG electrodes 
    (FP1, FP2, F3, F4, C3, C4, P3, P4, O1, O2, F7, F8, T3, T4, T5, T6, Fz, Cz, Pz) were placed according to the 10–20 International System,
    with odd indices referenced to the left and even indices to the right mastoid respectively, while the central electrodes (Fz, Cz, Pz) 
    were referenced to the half of the sum of the left and right mastoids. Signals’ sampling frequency was 200 Hz and a band pass filtered 
    at 0.5–40 Hz and notch filtered at 50 Hz were applied

Input: 1D array of EEG data typically Cz channel
       data is broeken up into windows of 1 second with 0.5 seconds overlap

"""


def signal_quality_index(eeg_1D,eeg_timePeriod,filename,dispIMG):

    "A semi-simulated EEG/EOG dataset for the comparison of EOG artifact rejection techniques"
    fs = 500
    mat = scipy.io.loadmat('/Users/joshuaighalo/Downloads/brainNet_datasets/semi_simulated dataset/Pure_Data.mat')
    #print(mat.keys())
    keys = list(mat.keys())[3:int(len(list(mat.keys())))]
    mdata = mat[keys[0]]
    dt = 1/fs
    stop = len(mdata.T)/fs
    Ts = (np.arange(0,stop,dt)).reshape(len(np.arange(0,stop,dt)),1)
    cz_CleanEEG = []
    for i in range(len(keys)):
        cz_CleanEEG.append((mat[keys[i]][17])[0:len(mdata.T)])
    cz_CleanEEG = [item for sublist in cz_CleanEEG for item in sublist]
    cz_CleanEEG = np.array(cz_CleanEEG)
        
    "Average Single-Sided Amplitude Spectrum (1-50Hz)"
    def amplitude_spectrum(data,sFreq):
        def param_fnc(data,sFreq):
            fft_vals = np.absolute(np.fft.rfft(data))
            fft_freq = np.fft.rfftfreq(len(data), 1.0/sFreq)
            freq_ix = np.where((fft_freq >= 1) & (fft_freq <= 50))[0]
            output = np.mean(fft_vals[freq_ix])
            return output
        ampSpectrum = []
        for i in range(len(data)):
            ampSpectrum.append(param_fnc(data[i],sFreq))
        ampSpectrum = np.array(ampSpectrum)
        return ampSpectrum


    "Line Noise - Average Single-Sided Amplitude Spectrum (59-61Hz range)"
    def line_noise(data,sFreq):
        def param_fnc(data,sFreq):
            fft_vals = np.absolute(np.fft.rfft(data))
            fft_freq = np.fft.rfftfreq(len(data), 1.0/sFreq)
            freq_ix = np.where((fft_freq >= 59) & (fft_freq <= 61))[0]
            output = np.mean(fft_vals[freq_ix])
            return output
        lineNoise = []
        for i in range(len(data)):
            lineNoise.append(param_fnc(data[i],sFreq))
        lineNoise = np.array(lineNoise)
        return lineNoise


    "RMS"
    def rms(data):
        def param_fnc(data):
            output = np.sqrt(np.mean(np.square(data)))
            return output
        rms = []
        for i in range(len(data)):
            rms.append(param_fnc(data[i]))
        rms = np.array(rms)
        return rms

    # Maximum Gradient
    def max_gradient(data):
        def param_fnc(data):
            output = np.max(np.diff(data))
            return output
        maxGradient = []
        for i in range(len(data)):
            maxGradient.append(param_fnc(data[i]))
        maxGradient = np.array(maxGradient)
        return maxGradient

    # Zero Crossing Rate
    def zero_crossing_rate(data):
        def param_fnc(data):
            output = np.mean(np.abs(np.diff(np.sign(data))))
            return output
        zcr = []
        for i in range(len(data)):
            zcr.append(param_fnc(data[i]))
        zcr = np.array(zcr)
        return zcr

    # Kurtosis
    def kurt(data):
        def param_fnc(data):
            output = kurtosis(data)
            return output
        kurtosis_ = []
        for i in range(len(data)):
            kurtosis_.append(param_fnc(data[i]))
        kurtosis_ = np.array(kurtosis_)
        return kurtosis_

    # Scoring formula
    def scoring(windows,mean,sd):
        def param_fnc(window,mean,sd):
            output=0
            if (window > mean - 1 * sd) and (window < mean + 1 * sd):
                output = 0
            elif (window < mean - 1 * sd) and (window > mean - 2 * sd):
                output = 1
            elif (window > mean + 1 * sd) and (window < mean + 2 * sd):
                output = 1
            elif (window < mean - 2 * sd) and (window > mean - 3 * sd):
                output = 2
            elif (window > mean + 2 * sd) and (window < mean + 3 * sd):
                output = 2
            elif (window < mean - 3 * sd) and (window > mean + 3 * sd):
                output = 3
            return output
        scores = []
        for i in range(len(windows)):
            scores.append(param_fnc(windows[i],mean,sd))
        scores = np.array(scores)
        return scores

    # Sliding Window
    def sliding_window(data_array,timing_array,window_size,step_size):
        """
        Inputs:
        1. data_array - 1D numpy array (d0 = channels) of data
        2. timing_array - 1D numpy array (d0 = samples) of timing data
        3. len(data_array) == len(timing_array)
        4. window_size - number of samples to use in each window in seconds e.g. 1 is 1 second
        5. step_size - the step size in seconds e.g.0.5 is 0.5 seconds 

        Outputs:    
        1. data_windows - 2D numpy array (d0 = windows, d1 = window size) of data

        """
        idx_winSize = np.where(timing_array == window_size)[0][0]
        idx_stepSize = np.where(timing_array == step_size)[0][0]
        shape = (data_array.shape[0] - idx_winSize + 1, idx_winSize)
        strides = (data_array.strides[0],) + data_array.strides
        rolled = np.lib.stride_tricks.as_strided(data_array, shape=shape, strides=strides)
        return rolled[np.arange(0,shape[0],idx_stepSize)]


    #split data into 1 second windows
    win_CleanEEG = sliding_window(cz_CleanEEG,Ts,1.0,0.5)

    ampSpec_cleanEEG = amplitude_spectrum(win_CleanEEG,fs)
    lineNoise_cleanEEG = line_noise(win_CleanEEG,fs)
    rms_cleanEEG = rms(win_CleanEEG)
    maxGrad_cleanEEG = max_gradient(win_CleanEEG)
    zcr_cleanEEG = zero_crossing_rate(win_CleanEEG)
    kurt_cleanEEG = kurt(win_CleanEEG)

    mean_ampSpec_cleanEEG = np.mean(ampSpec_cleanEEG)
    std_ampSpec_cleanEEG = np.std(ampSpec_cleanEEG)
    mean_lineNoise_cleanEEG = np.mean(lineNoise_cleanEEG)
    std_lineNoise_cleanEEG = np.std(lineNoise_cleanEEG)
    mean_rms_cleanEEG = np.mean(rms_cleanEEG)
    std_rms_cleanEEG = np.std(rms_cleanEEG)
    mean_maxGrad_cleanEEG = np.mean(maxGrad_cleanEEG)
    std_maxGrad_cleanEEG = np.std(maxGrad_cleanEEG)
    mean_zcr_cleanEEG = np.mean(zcr_cleanEEG)
    std_zcr_cleanEEG = np.std(zcr_cleanEEG)
    mean_kurt_cleanEEG = np.mean(kurt_cleanEEG)
    std_kurt_cleanEEG = np.std(kurt_cleanEEG)

    # segment the input eeg data into windows of 1 second
    input_data = sliding_window(eeg_1D,eeg_timePeriod,1.0,0.5)
    input_ampSpec = amplitude_spectrum(input_data,fs)
    input_lineNoise = line_noise(input_data,fs)
    input_rms = rms(input_data)
    input_maxGrad = max_gradient(input_data)
    input_zcr = zero_crossing_rate(input_data)
    input_kurt = kurt(input_data)

    zscore_ampSpec = scoring(input_ampSpec,mean_ampSpec_cleanEEG,std_ampSpec_cleanEEG)
    zscore_ampSpec = zscore_ampSpec.reshape(len(zscore_ampSpec),1)
    clean_ampSpec = int((np.count_nonzero(zscore_ampSpec==0)/len(zscore_ampSpec))*100)
    zscore_lineNoise = scoring(input_lineNoise,mean_lineNoise_cleanEEG,std_lineNoise_cleanEEG)
    zscore_lineNoise = zscore_lineNoise.reshape(len(zscore_lineNoise),1)
    clean_lineNoise = int((np.count_nonzero(zscore_lineNoise==0)/len(zscore_lineNoise))*100)
    zscore_rms = scoring(input_rms,mean_rms_cleanEEG,std_rms_cleanEEG)
    zscore_rms = zscore_rms.reshape(len(zscore_rms),1)
    clean_rms = int((np.count_nonzero(zscore_rms==0)/len(zscore_rms))*100)
    zscore_maxGrad = scoring(input_maxGrad,mean_maxGrad_cleanEEG,std_maxGrad_cleanEEG)
    zscore_maxGrad = zscore_maxGrad.reshape(len(zscore_maxGrad),1)
    clean_maxGrad = int((np.count_nonzero(zscore_maxGrad==0)/len(zscore_maxGrad))*100)
    zscore_zcr = scoring(input_zcr,mean_zcr_cleanEEG,std_zcr_cleanEEG)
    zscore_zcr = zscore_zcr.reshape(len(zscore_zcr),1)
    clean_zcr = int((np.count_nonzero(zscore_zcr==0)/len(zscore_zcr))*100)
    zscore_kurt = scoring(input_kurt,mean_kurt_cleanEEG,std_kurt_cleanEEG)
    zscore_kurt = zscore_kurt.reshape(len(zscore_kurt),1)
    clean_kurt = int((np.count_nonzero(zscore_kurt==0)/len(zscore_kurt))*100)

    total = np.mean(np.hstack((zscore_ampSpec,zscore_lineNoise,zscore_rms,zscore_maxGrad,zscore_zcr,zscore_kurt)),axis=1)
    quality = int((np.count_nonzero(total==0)/len(total))*100)

    if dispIMG == True:
        print(filename+'\n'+'Amplitude Spectrum: '+str(clean_ampSpec)+'%\n'+'Line Noise: '+str(clean_lineNoise)+'%\n'+'RMS: '+str(clean_rms)+'%\n'+'Maximum Gradient: '+str(clean_maxGrad)+'%\n'+'ZCR: '+str(clean_zcr)+'%\n'+'Kurtosis: '+str(clean_kurt)+'%\n'+'Signal Quality: '+str(quality)+'%')
    else:
        pass
    
    print('\n')
    return quality