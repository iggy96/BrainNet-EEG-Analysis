# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 19:00:13 2021

@author: oseho
"""


from eeg_libs import*
from params import*
import params as cfg


# Preprocessing functions
# Event Related Potentials (ERP) extracting functions 
class importFile:
    """
    Functionality is geared towards importing raw eeg files from the gtec system.
    The class contains a subclass which is the nuerocatch class
        -   The neurocatch class facilitates the selection of the version (either V1.0 or v1.1) of the neurocatch system
            utilized during the recording of the eeg data.
        -   This function also contains an EOG channel selector block which facilitates the correct selection of the 
            EOG channel.
        -   Function returns the raw EEG,raw EOG,an array holding both raw EEG and raw EOG,time (Ts) and the trigger
            channel (trig)
    """
    class neurocatch:
        def init(self,version,filename,localPath,dispIMG):
            if version == 1.0:
                data = filename
                localPath = localPath.replace(os.sep, '/')   
                localPath = localPath + '/'  
                path = localPath+data
                os.chdir(path)
                for file in (os.listdir(path)):
                    if file.endswith(".txt"):
                        file_path = f"{path}/{file}"
                        
                with open(file_path, 'r') as f:
                    lines = f.readlines(200)

                def two_digits(data):
                    # electrodes with Nan kOhms 
                    reject_impedance = 1000
                    fullstring = data
                    substring = "NaN"
                    if substring in fullstring:
                        val3 = reject_impedance
                    # electrodes with numeric kOhms
                    else:
                        val1 = data
                        val2 = val1[4:] # delete unnecessary characters from the front
                        val2 = val2[:2]
                        val3 = int(float(val2)) # delete unnecessary characters from the back then convert to integer
                        import math
                        # check 1
                        digits = int(math.log10(val3))+1 # check number of digits in result
                        if digits == 2: # expected result
                            val3 = val3
                        if digits < 2: # unexpected result 1
                            val5 = val1
                            val5 = val5[4:]
                            val5 = val5[:3]
                            val3 = int(float(val5))
                        # check 2
                        digits = int(math.log10(val3))+1 # check number of digits in result
                        if digits == 2: # expected result
                            val3 = val3
                        if digits < 1: # unexpected result 1
                            val6 = val1
                            val6 = val6[4:]
                            val6 = val6[:4]
                            val3 = int(float(val6))
                    return val3
                    
                def three_digits(data):
                    # electrodes with Nan kOhms 
                    reject_impedance = 1000
                    fullstring = data
                    substring = "NaN"
                    if substring in fullstring:
                        val3 = reject_impedance
                    # electrodes with numeric kOhms
                    else:
                        val1 = data
                        val2 = val1[4:]
                        val2 = val2[:3]
                        val3 = int(float(val2))
                        import math
                        # check 1
                        digits = int(math.log10(val3))+1 # check number of digits in result
                        if digits == 3: # expected result
                            val3 = val3
                        if digits < 3: # unexpected result 1
                            val5 = val1
                            val5 = val5[4:]
                            val5 = val5[:4]
                            val3 = int(float(val5))
                        # check 2
                        digits = int(math.log10(val3))+1 # check number of digits in result
                        if digits == 3: # expected result
                            val3 = val3
                        if digits < 2: # unexpected result 1
                            val6 = val1
                            val6 = val6[4:]
                            val6 = val6[:5]
                            val3 = int(float(val6))
                    return val3
                    
                # extract from metadata file, channels that collected eeg data
                device_chans = ['FZ','CZ','P3','PZ','P4','PO7','PO8','OZ','unknown channel','unknown channel','unknown channel','sampleNumber','battery','trigger']
                def lcontains(needle_s, haystack_l):
                    try: return [i for i in haystack_l if needle_s in i][0]
                    except IndexError: return None
                metadata_chans = [lcontains(device_chans[i],lines) for i in range(len(device_chans))]

                p3 = metadata_chans[2]
                if len(p3)<=15:
                    p3 = two_digits(p3)
                else:
                    p3 = three_digits(p3)
                p4 = metadata_chans[4]
                if len(p4)<=15:
                    p4 = two_digits(p4)
                else:
                    p4 = three_digits(p4)

                p07 = metadata_chans[5]
                if len(p07)<=15:
                    p07 = two_digits(p07)
                else:
                    p07 = three_digits(p07)

                p08 = metadata_chans[6]
                if len(p08)<=15:
                    p08 = two_digits(p08)
                else:
                    p08 = three_digits(p08)

                oz = metadata_chans[7]
                if len(oz)<=15:
                    oz = two_digits(oz)
                else:
                    oz = three_digits(oz)

            elif version == 1.1:
                localPath = localPath.replace(os.sep, '/')   
                localPath = localPath + '/'  
                path = localPath+filename
                os.chdir(path)
                for file in os.listdir():
                    if file.endswith(".json"):
                        file_path = f"{path}/{file}"

                metadata = open(file_path)
                metadata = json.load(metadata)
                if dispIMG == True:
                    for i in metadata:
                        print(i)
                    print(metadata['version'])
                else:
                    pass
                metadata_chans = metadata['channels']
                metadata_imp = metadata['impedances']
                # p3,p4,p07,p08,oz
                p3 = (metadata_imp[0])['P3']
                p4 = (metadata_imp[0])['P4']
                p07 = (metadata_imp[0])['PO7']
                p08 = (metadata_imp[0])['PO8']
                oz = (metadata_imp[0])['OZ']

            #  import raw file 1
            pathBin = [path+'/'+filename+'.bin']
            filenames = glob.glob(pathBin[0])
            if dispIMG == True:
                print(filenames)
            else:
                pass
            data = [np.fromfile(f, dtype=np.float32) for f in filenames]
            data1 = data[0]
            dataCols = len(metadata_chans)
            dataRows = int(len(data1)/dataCols)           
            data1 = data1.reshape(dataRows, dataCols)

            #  configure eeg channels
            eegChans = gtec['eegChans']
            fz = eegChans[0]
            fz1 = data1[:,fz]     #extract column 0
            fz = fz1.reshape(1,dataRows)

            cz = eegChans[1]
            cz1 = data1[:,cz]
            cz = cz1.reshape(1, dataRows)

            pz = eegChans[2]
            pz1 = data1[:,pz]
            pz = pz1.reshape(1,dataRows)

            # %% configure eog channels
            if p3 < 501:
                eogChans = gtec['eogChans']
                eog10 = eogChans[0]
                eogChan1 = data1[:,eog10]
                eogChan1 = eogChan1.reshape(1,dataRows)
                if dispIMG == True:
                    print('channel P3 utilized')
                else:
                    pass
            else:
                eogChan1 = np.zeros(len(fz.T))
                eogChan1 = eogChan1.reshape(1,len(eogChan1))

            if p4 < 501:
                eogChans = gtec['eogChans']
                eog20 = eogChans[1]
                eogChan2 = data1[:,eog20]
                eogChan2 = eogChan2.reshape(1,dataRows)
                if dispIMG == True:
                    print('channel P4 utilized')
                else:
                    pass
            else:
                eogChan2 = np.zeros(len(fz.T))
                eogChan2 = eogChan2.reshape(1,len(eogChan2))

            if p07 < 501:
                eogChans = gtec['eogChans']
                eog30 = eogChans[2]
                eogChan3 = data1[:,eog30]
                eogChan3 = eogChan3.reshape(1,dataRows)
                if dispIMG == True:
                    print('channel P07 utilized')
                else:
                    pass
            else:
                eogChan3 = np.zeros(len(fz.T))
                eogChan3 = eogChan3.reshape(1,len(eogChan3))

            if p08 < 501:
                eogChans = gtec['eogChans']
                eog40 = eogChans[3]
                eogChan4 = data1[:,eog40]
                eogChan4 = eogChan4.reshape(1,dataRows)
                if dispIMG == True:
                    print('channel P08 utilized')
                else:
                    pass
            else:
                eogChan4 = np.zeros(len(fz.T))
                eogChan4 = eogChan4.reshape(1,len(eogChan4))

            if oz < 501:
                eogChans = gtec['eogChans']
                eog50 = eogChans[4]
                eogChan5 = data1[:,eog50]
                eogChan5 = eogChan5.reshape(1,dataRows)
                if dispIMG == True:
                    print('channel 0Z utilized')
                else:
                    pass
            else:
                eogChan5 = np.zeros(len(fz.T))
                eogChan5 = eogChan5.reshape(1,len(eogChan5))

            # %% configure trigger channel
            trigCol = gtec['trigCol']
            trig = data1[:,trigCol]
            trig = trig.reshape(1,dataRows)

            # %% configure raw file
            rawData = np.concatenate((fz, cz, pz,eogChan1,eogChan2,eogChan3,eogChan4,eogChan5,trig))
            rawData = rawData.T

            # delete non zero columns i.e., the eogchans that are not in the data represented by zero columns
            mask = (rawData == 0).all(0)
                # Find the indices of these columns
            column_indices = np.where(mask)[0]
                # Update x to only include the columns where non-zero values occur.
            rawData = rawData[:,~mask] # rawData containing eegChans,

            # %% the new raw data just containing the required eegchans,eogchans and the trig channel
                # correctly name channels in raw Data
            csm = dict(fz=[0], cz=[1], pz=[2], eog1=[3], eog2=[4], eog3=[5], eog4=[6],eog5=[7], ntrig=[8])
            csm_fz = csm['fz']
            csm_fz = csm_fz[0]
            csm_cz = csm['cz']
            csm_cz = csm_cz[0]
            csm_pz = csm['pz']
            csm_pz = csm_pz[0]
            csm_eog1 = csm['eog1']
            csm_eog1 = csm_eog1[0]
            csm_eog2 = csm['eog2']
            csm_eog2 = csm_eog2[0]
            csm_eog3 = csm['eog3']
            csm_eog3 = csm_eog3[0]
            csm_eog4 = csm['eog4']
            csm_eog4 = csm_eog4[0]
            csm_eog5 = csm['eog5']
            csm_eog5 = csm_eog5[0]
            csm_ntrig = csm['ntrig']
            csm_ntrig = csm_ntrig[0]

            if len(rawData.T)==4:
                csm_ntrig = 3
                rawData = rawData[:,[csm_fz,csm_cz,csm_pz,csm_ntrig]]  
                rawEEG = rawData[:,[csm_fz,csm_cz,csm_pz]]              
                rawEOG = rawData[:,[csm_ntrig]]
                rawEEGEOG = np.concatenate((rawEEG,rawEOG),axis=1)
                if dispIMG == True:
                    print('data contains Fz, Cz, Pz & no EOG channels')
                else:
                    pass
            
            elif len(rawData.T)==5:
                csm_ntrig = 4
                rawData = rawData[:,[csm_fz,csm_cz,csm_pz,csm_eog1,csm_ntrig]]  
                rawEEG = rawData[:,[csm_fz,csm_cz,csm_pz]]              
                rawEOG = rawData[:,[csm_eog1]]
                rawEEGEOG = np.concatenate((rawEEG,rawEOG),axis=1)
                if dispIMG == True:
                    print('data contains Fz, Cz, Pz & one EOG channel')
                else:
                    pass

            elif len(rawData.T)==6:
                csm_ntrig = 5
                rawData = rawData[:,[csm_fz,csm_cz,csm_pz,csm_eog1,csm_eog2,csm_ntrig]]  
                rawEEG = rawData[:,[csm_fz,csm_cz,csm_pz]]              
                rawEOG = rawData[:,[csm_eog1,csm_eog2]]
                rawEEGEOG = np.concatenate((rawEEG,rawEOG),axis=1)
                if dispIMG == True:
                    print('data contains Fz, Cz, Pz & two EOG channels')
                else:
                    pass

            elif len(rawData.T)==7:
                csm_ntrig = 6
                rawData = rawData[:,[csm_fz,csm_cz,csm_pz,csm_eog1,csm_eog2,csm_eog3,csm_ntrig]]  
                rawEEG = rawData[:,[csm_fz,csm_cz,csm_pz]]              
                rawEOG = rawData[:,[csm_eog1,csm_eog2,csm_eog3]]
                rawEEGEOG = np.concatenate((rawEEG,rawEOG),axis=1)
                if dispIMG == True:
                    print('data contains Fz, Cz, Pz & three EOG channels')
                else:
                    pass

            elif len(rawData.T)==8:
                csm_ntrig = 7
                rawData = rawData[:,[csm_fz,csm_cz,csm_pz,csm_eog1,csm_eog2,csm_eog3,csm_eog4,csm_ntrig]]  
                rawEEG = rawData[:,[csm_fz,csm_cz,csm_pz]]              
                rawEOG = rawData[:,[csm_eog1,csm_eog2,csm_eog3,csm_eog4]]
                rawEEGEOG = np.concatenate((rawEEG,rawEOG),axis=1)
                if dispIMG == True:
                    print('data contains Fz, Cz, Pz & four EOG channels')
                else:
                    pass

            elif len(rawData.T)==9:
                csm_ntrig = 8
                rawData = rawData[:,[csm_fz,csm_cz,csm_pz,csm_eog1,csm_eog2,csm_eog3,csm_eog4,csm_eog5,csm_ntrig]]  
                rawEEG = rawData[:,[csm_fz,csm_cz,csm_pz]]              
                rawEOG = rawData[:,[csm_eog1,csm_eog2,csm_eog3,csm_eog4,csm_eog5]]
                rawEEGEOG = np.concatenate((rawEEG,rawEOG),axis=1)
                if dispIMG == True:
                    print('data contains Fz, Cz, Pz & five EOG channels')
                else:
                    pass

            # time period of scan
            fs = gtec['fs']
            dt = 1/fs
            stop = dataRows/fs
            Ts = (np.arange(0,stop,dt)).reshape(len(np.arange(0,stop,dt)),1)
            return rawEEG,rawEOG,rawEEGEOG,Ts,trig

class filters:
    """
     filters for EEG data
     filtering order: adaptive filter -> notch filter -> bandpass filter (or lowpass filter, highpass filter)
    """
    def notch(self,data,line,fs,Q=30):
        """
           Inputs  :   data    - 2D numpy array (d0 = samples, d1 = channels) of unfiltered EEG data
                       cut     - frequency to be notched (defaults to config)
                       fs      - sampling rate of hardware (defaults to config)
                       Q       - Quality Factor (defaults to 30) that characterizes notch filter -3 dB bandwidth bw relative to its center frequency, Q = w0/bw.   
           Output  :   y     - 2D numpy array (d0 = samples, d1 = channels) of notch-filtered EEG data
           NOTES   :   
           Todo    : report testing filter characteristics
        """
        cut = line
        w0 = cut/(fs/2)
        b, a = signal.iirnotch(w0, Q)
        y = signal.filtfilt(b, a, data, axis=0)
        return y

    def butterBandPass(self,data,lowcut,highcut,fs,order=4):
        """
           Inputs  :   data    - 2D numpy array (d0 = samples, d1 = channels) of unfiltered EEG data
                       low     - lower limit in Hz for the bandpass filter (defaults to config)
                       high    - upper limit in Hz for the bandpass filter (defaults to config)
                       fs      - sampling rate of hardware (defaults to config)
                       order   - the order of the filter (defaults to 4)  
           Output  :   y     - 2D numpy array (d0 = samples, d1 = channels) of notch-filtered EEG data
           NOTES   :   
           Todo    : report testing filter characteristics
         data: eeg data (samples, channels)
         some channels might be eog channels
        """
        low_n = lowcut
        high_n = highcut
        sos = butter(order, [low_n, high_n], btype="bandpass", analog=False, output="sos",fs=fs)
        y = sosfiltfilt(sos, data, axis=0)
        return y

    def adaptive(self,eegData,eogData,nKernel=5, forgetF=0.995,  startSample=0, p = False):
        """
           Inputs:
           eegData - A matrix containing the EEG data to be filtered here each channel is a column in the matrix, and time
           starts at the top row of the matrix. i.e. size(data) = [numSamples,numChannels]
           eogData - A matrix containing the EOG data to be used in the adaptive filter
           startSample - the number of samples to skip for the calculation (i.e. to avoid the transient)
           p - plot AF response (default false)
           nKernel = Dimension of the kernel for the adaptive filter
           Outputs:
           cleanData - A matrix of the same size as "eegdata", now containing EOG-corrected EEG data.
           Adapted from He, Ping, G. Wilson, and C. Russell. "Removal of ocular artifacts from electro-encephalogram by adaptive filtering." Medical and biological engineering and computing 42.3 (2004): 407-412.
        """
        #   reshape eog array if necessary
        if len(eogData.shape) == 1:
            eogData = np.reshape(eogData, (eogData.shape[0], 1))
        # initialise Recursive Least Squares (RLS) filter state
        nEOG = eogData.shape[1]
        nEEG = eegData.shape[1]
        hist = np.zeros((nEOG, nKernel))
        R_n = np.identity(nEOG * nKernel) / 0.01
        H_n = np.zeros((nEOG * nKernel, nEEG))
        X = np.hstack((eegData, eogData)).T          # sort EEG and EOG channels, then transpose into row variables
        eegIndex = np.arange(nEEG)                              # index of EEG channels within X
        eogIndex = np.arange(nEOG) + eegIndex[-1] + 1           # index of EOG channels within X
        for n in range(startSample, X.shape[1]):
            hist = np.hstack((hist[:, 1:], X[eogIndex, n].reshape((nEOG, 1))))  # update the EOG history by feeding in a new sample
            tmp = hist.T                                                        # make it a column variable again (?)
            r_n = np.vstack(np.hsplit(tmp, tmp.shape[-1]))
            K_n = np.dot(R_n, r_n) / (forgetF + np.dot(np.dot(r_n.T, R_n), r_n))                                           # Eq. 25
            R_n = np.dot(np.power(forgetF, -1),R_n) - np.dot(np.dot(np.dot(np.power(forgetF, -1), K_n), r_n.T), R_n)       #Update R_n
            s_n = X[eegIndex, n].reshape((nEEG, 1))                   #get EEG signal and make sure it's a 1D column array
            e_nn = s_n - np.dot(r_n.T, H_n).T  #Eq. 27
            H_n = H_n + np.dot(K_n, e_nn.T)
            e_n = s_n - np.dot(r_n.T, H_n).T
            X[eegIndex, n] = np.squeeze(e_n)
        cleanData = X[eegIndex, :].T
        return cleanData

    def butter_lowpass(self,data,cutoff,fs):

        def params(data,cutoff,fs,order=4):
            nyq = 0.5 * fs
            normal_cutoff = cutoff / nyq
            b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
            y = signal.lfilter(b, a, data)
            return y
        output = []
        for i in range(len(data.T)):
            output.append(params(data[:,i],cutoff,fs))
        return np.array(output).T

    def butter_highpass(self,data,cutoff,fs,order=4):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
        y = signal.filtfilt(b, a, data)
        return y

def rising_edge(data):
    # used in trigger channel development before epoching
    trig = data
    trg = (trig >= 1) & (trig < 9)
    trig[trg > 1] = 0     # cut off any onset offset triggers outside the normal range.
    diff = np.diff(trig)
    t = diff > 0  # rising edges assume true
    k = np.array([ False])
    k = k.reshape(1,len(k))
    pos = np.concatenate((k, t),axis=1)
    trig[~pos] = 0
    newtrig = trig 
    trigCol = newtrig
    trigCol = trigCol.T
    trigCol = trigCol.reshape(len(trigCol))
    return trigCol

def peaktopeak(data):
    # used for artifact rejection
    a = np.amax(data, axis = 1)
    b = np.amin(data, axis = 1)
    p2p = a-b
    return p2p

class erpExtraction:
    """
      Inputs: trigger data produced from rising_edge()
            : standard and deviant tones elicit the N100 and P300 erps
            : congruent and incongruent words elicit the N400 erps
      Outputs: ERP data:    N100 & P300: ERPs (channels,stimulusERP,length of ERP), epochs (channels,stimulusERP), latency (1D array)
                            N400: ERPs (channels,stimulusERP,length of ERP), epochs (channels,stimulusERP), latency (1D array)
      Notes:   - The trigger channel is assumed to be the last channel in the data matrix
                - stimulusERP (standard tones ERP, deviant tones ERP, congruent words ERP, incongruent words ERP)
                - epochs (ar_stimuli) are padded with nans to reach a length of 300 samples for cases of epoch extraction 
    """
    def N100P300(self,eegData,period,trigger_channel,scanID,chanNames,stimTrig,clip,lowpass,fs,dispIMG):
        """
          Inputs: trigger channels, bandpass filtered data for all channels, time period,
                  stimTrig, clip value
        """
        def addNan(x):
            noEpochs = 300
            lenRow = len(x)
            lenCol = len(x.T)
            remainRows = int(noEpochs-lenRow)
            nansValues = np.full((remainRows,lenCol),np.nan)
            return np.vstack((x,nansValues))

        def N100P300_(eegData,period,trigger_channel,scanID,chanNames,stimTrig,clip,lowpass,fs,dispIMG):
            chanNames,scanID,trigger_channel,channel_data,period,stimTrig,clip,lowpass,fs,dispIMG = chanNames,scanID,trigger_channel,eegData,period,stimTrig,clip,lowpass,fs,dispIMG
            def algorithm(chanNames,scanID,trigger_channel,channel_data,period,stimTrig,clip,lowpass,fs,dispIMG):
                trigger_data = rising_edge(trigger_channel)
                trigCol = trigger_data
                avg = channel_data 
                Ts = period
                # STANDARD TONE [1]: extract the time points where stimuli 1 exists
                std = stimTrig['std']
                std = std[0]
                
                no_ones = np.count_nonzero(trigCol==1)
                if dispIMG == True:
                    print("number of std tone event codes:",no_ones)
                else:
                    pass
                
                result = np.where(trigCol == std)
                idx_10 = result[0]
                idx_11 = idx_10.reshape(1,len(idx_10))
                
                # sort target values
                desiredCols = trigCol[idx_10]
                desiredCols = desiredCols.reshape(len(desiredCols),1)
                
                # sort values before target
                idx_st = (idx_11 - 50)
                i = len(idx_st.T)
                allArrays = np.array([])
                for x in range(i):
                    myArray = trigCol[int(idx_st[:,x]):int(idx_11[:,x])]
                    allArrays = np.concatenate([allArrays,myArray])
                startVals = allArrays
                startCols = startVals.reshape(no_ones,50)
                
                # sort values immediately after target
                idx_en11 = idx_11 + 1
                postTargetCols = trigCol[idx_en11]
                postTargetCols = postTargetCols.T
                
                # sort end values
                    # ----------------------------------------------------------------------------
                    # determination of the number steps to reach the last point of each epoch
                    # the event codes are not evenly timed hence the need for steps determination
                a = trigCol[idx_10]
                a = a.reshape(len(a),1)
                
                b = Ts[idx_10]
                
                c_i = float("%0.3f" % (Ts[int(len(Ts)-1)]))
                c = (np.array([0,c_i]))
                c = c.reshape(1,len(c))
                
                std_distr = np.concatenate((a, b),axis=1)
                std_distr = np.vstack((std_distr,c))
                
                std_diff = np.diff(std_distr[:,1],axis=0)
                std_step = ((std_diff/0.002)-1)
                std_step = np.where(std_step > 447, 447, std_step)
                
                # check if the number of steps are the same if yes, meaning the epochs have the same length
                result = np.all(std_step == std_step[0])
                if result:
                    # use for equal epoch steps
                    # sort end values
                    idx_en12 = idx_en11 + std_step.T
                    i = len(idx_en12.T)
                    allArrays = np.array([])
                    for x in range(i):
                        myArray = trigCol[int(idx_en11[:,x]):int(idx_en12[:,x])]
                        allArrays = np.concatenate([allArrays,myArray])
                    endVals = allArrays
                    endCols = endVals.reshape(no_ones,447)
                    
                    # merge the different sections of the epoch to form the epochs
                    trig_std = np.concatenate((startCols,desiredCols,endCols),axis=1)
                
                else:
                    # use when we have unequal epoch steps
                        # --------------------------------------------
                        # apply the step to get the index of the last point of the epoch
                    idx_en12 = idx_en11 + std_step.T 
                    i = len(idx_en12.T)
                    allArrays = []
                    for x in range(i):
                        myArray = trigCol[int(idx_en11[:,x]):int(idx_en12[:,x])]
                        allArrays.append(myArray)
                    endVals = allArrays
                        # add zeros to fill enable epochs of shorter length equal with epochs of adequate length
                    endVals = pd.DataFrame(endVals)
                    endVals.fillna(endVals.mean(),inplace=True)
                    endCols = endVals.values
                    
                    # merge the different sections of the epoch to form the epochs
                    trig_std = np.concatenate((startCols,desiredCols,endCols),axis=1)
                
                # ----------------------------------------------------------------------------------------------------------------
                # implement trig epoch creation to eeg data
                    # replace trigCol with avg 
                    # get start, desired, post and end col, then merge together
                    # remove data equal to non 1 or o triggers
                
                # sort target values
                eeg_desiredCols = avg[idx_10]
                eeg_desiredCols = eeg_desiredCols.reshape(len(eeg_desiredCols),1)
                
                # sort values before target
                i = len(idx_st.T)
                allArrays = np.array([])
                for x in range(i):
                    myArray = avg[int(idx_st[:,x]):int(idx_11[:,x])]
                    allArrays = np.concatenate([allArrays,myArray])
                eeg_startVals = allArrays
                eeg_startCols = eeg_startVals.reshape(no_ones,50)
                
                # sort values immediately after target
                eeg_postTargetCols = avg[idx_en11]
                eeg_postTargetCols = eeg_postTargetCols.T
                
                # check if the number of steps are the same if yes, meaning the epochs have the same length
                result = np.all(std_step == std_step[0])
                if result:
                    # use for equal epochs
                    # sort end values
                    idx_en12 = idx_en11 + std_step.T
                    i = len(idx_en12.T)
                    allArrays = np.array([])
                    for x in range(i):
                        myArray = avg[int(idx_en11[:,x]):int(idx_en12[:,x])]
                        allArrays = np.concatenate([allArrays,myArray])
                    endVals = allArrays
                    eeg_endCols = endVals.reshape(no_ones,447)
                    
                    # merge the different sections of the epoch to form the epochs
                    # eeg_std = np.concatenate((eeg_startCols,eeg_desiredCols,eeg_endCols),axis=1)
                    epochs_std = np.concatenate((eeg_startCols,eeg_desiredCols,eeg_endCols),axis=1)
                    
                else:
                    # use for unequal epochs
                    # sort end values
                    idx_en12 = idx_en11 + std_step.T 
                    i = len(idx_en12.T)
                    allArrays = []
                    for x in range(i):
                        myArray = avg[int(idx_en11[:,x]):int(idx_en12[:,x])]
                        allArrays.append(myArray)
                    eeg_endVals = allArrays

                        # add zeros to fill enable epochs of shorter length equal with epochs of adequate length
                    eeg_endVals = pd.DataFrame(eeg_endVals)
                    eeg_endVals.fillna(eeg_endVals.mean(),inplace=True)
                    eeg_endCols = eeg_endVals.values
                    
                    # merge the different sections of the epoch to form the epochs
                    epochs_std = np.concatenate((eeg_startCols,eeg_desiredCols,eeg_endCols),axis=1)
                
                # baseline correction
                prestim = epochs_std[:,0:49]
                mean_prestim = np.mean(prestim,axis=1)
                mean_prestim = mean_prestim.reshape(len(mean_prestim),1)
                bc_std = epochs_std - mean_prestim
                
                # artefact rejection
                p2p = peaktopeak(bc_std)
                result = np.where(p2p > clip)
                row = result[0]
                ar_std = np.delete(bc_std,(row),axis = 0)
                filtering = filters()
                ar_std = filtering.butter_lowpass(ar_std.T,lowpass,fs)
                ar_std = ar_std.T
                dif = ((len(bc_std)-len(ar_std))/len(bc_std))*100
                if len(ar_std) == len(bc_std):
                    if dispIMG == True:
                        print("notice! epochs lost for std tone:","{:.2%}".format((int(dif))/100))
                    else:
                        pass
                elif len(ar_std) < len(bc_std):
                    if dispIMG == True:
                        print("callback! epochs lost for std tone:","{:.2%}".format((int(dif))/100))
                    else:
                        pass
                
                if ar_std.size == 0:
                    epochLen = bc_std.shape[1]
                    noEpochs = 300
                    ar_std = np.full((noEpochs,epochLen),np.nan)
                    avg_std = np.nanmean(ar_std,axis=0)
                    print(chanNames,':',scanID," removed for standard tones N1P3 analysis as all its epochs exceed the clip value of",clip)
                elif ar_std.size != 0:
                    # averaging
                    ar_std = addNan(ar_std)
                    avg_std = np.nanmean(ar_std,axis=0)
                avg_std = avg_std

                #%%
                # DEVIANT TONE [2]: extract the time points where stimuli 2 exists
                dev = stimTrig['dev']
                dev = dev[0]
                
                no_twos = np.count_nonzero(trigCol==2)
                if dispIMG == True:
                    print("number of dev tone event codes:",no_twos)
                else:
                    pass
                
                result = np.where(trigCol == dev)
                idx_20 = result[0]
                idx_21 = idx_20.reshape(1,len(idx_20))
                
                # sort target values
                desiredCols = trigCol[idx_20]
                desiredCols = desiredCols.reshape(len(desiredCols),1)
                
                # sort values before target
                idx_dev = (idx_21 - 50)
                i = len(idx_dev.T)
                allArrays = np.array([])
                for x in range(i):
                    myArray = trigCol[int(idx_dev[:,x]):int(idx_21[:,x])]
                    allArrays = np.concatenate([allArrays,myArray])
                startVals = allArrays
                startCols = startVals.reshape(no_twos,50)
                
                # sort values immediately after target
                idx_en21 = idx_21 + 1
                postTargetCols = trigCol[idx_en21]
                postTargetCols = postTargetCols.T
                
                # sort end values
                    # ----------------------------------------------------------------------------
                    # determination of the number steps to reach the last point of each epoch
                    # the event codes are not evenly timed hence the need for steps determination
                a = trigCol[idx_20]
                a = a.reshape(len(a),1)
                
                b = Ts[idx_20]
                
                c_i = float("%0.3f" % (Ts[int(len(Ts)-1)]))
                c = (np.array([0,c_i]))
                c = c.reshape(1,len(c))
                
                dev_distr = np.concatenate((a, b),axis=1)
                dev_distr = np.vstack((dev_distr,c))
                
                dev_diff = np.diff(dev_distr[:,1],axis=0)
                dev_step = ((dev_diff/0.002)-1)
                dev_step = np.where(dev_step > 447, 447, dev_step)
                
                # check if the number of steps are the same if yes, meaning the epochs have the same length
                result = np.all(dev_step == dev_step[0])
                if result:
                    # sort end values
                    idx_en22 = idx_en21 + dev_step.T
                    i = len(idx_en22.T)
                    allArrays = np.array([])
                    for x in range(i):
                        myArray = trigCol[int(idx_en21[:,x]):int(idx_en22[:,x])]
                        allArrays = np.concatenate([allArrays,myArray])
                    endVals = allArrays
                    endCols = endVals.reshape(no_twos,447)
                
                        # merge the different sections of the epoch to form the epochs
                    trig_dev = np.concatenate((startCols,desiredCols,endCols),axis=1)
                else:
                    # use when we have unequal epoch steps
                        # apply the step to get the index of the last point of the epoch
                    idx_en22 = idx_en21 + dev_step.T 
                    i = len(idx_en22.T)
                    allArrays = []
                    for x in range(i):
                        myArray = trigCol[int(idx_en21[:,x]):int(idx_en22[:,x])]
                        allArrays.append(myArray)
                    endVals = allArrays
                        # add zeros to fill enable epochs of shorter length equal with epochs of adequate length
                    endVals = pd.DataFrame(endVals)
                    endVals.fillna(endVals.mean(),inplace=True)
                    endCols = endVals.values
                    
                        # merge the different sections of the epoch to form the epochs
                    trig_devpad = np.concatenate((startCols,desiredCols,endCols),axis=1)
                
                # ----------------------------------------------------------------------------------------------------------------
                # implement trig epoch creation to eeg data
                    # replace trigCol with avg 
                    # get start, desired, post and end col, then merge together
                    # remove data equal to non 1 or o triggers
                
                # sort target values
                eeg_desiredCols = avg[idx_20]
                eeg_desiredCols = eeg_desiredCols.reshape(len(eeg_desiredCols),1)
                
                # sort values before target
                i = len(idx_dev.T)
                allArrays = np.array([])
                for x in range(i):
                    myArray = avg[int(idx_dev[:,x]):int(idx_21[:,x])]
                    allArrays = np.concatenate([allArrays,myArray])
                eeg_startVals = allArrays
                eeg_startCols = eeg_startVals.reshape(no_twos,50)
                
                # sort values immediately after target
                eeg_postTargetCols = avg[idx_en21]
                eeg_postTargetCols = eeg_postTargetCols.T
                
                # check if the number of steps are the same if yes, meaning the epochs have the same length
                result = np.all(dev_step == dev_step[0])
                if result:
                    # use for equal epochs
                    # sort end values
                    idx_en22 = idx_en21 + dev_step.T
                    i = len(idx_en22.T)
                    allArrays = np.array([])
                    for x in range(i):
                        myArray = avg[int(idx_en21[:,x]):int(idx_en22[:,x])]
                        allArrays = np.concatenate([allArrays,myArray])
                    endVals = allArrays
                    eeg_endCols = endVals.reshape(no_twos,447)
                    
                    # merge the different sections of the epoch to form the epochs
                    # eeg_dev = np.concatenate((eeg_startCols,eeg_desiredCols,eeg_endCols),axis=1)
                    epochs_dev = np.concatenate((eeg_startCols,eeg_desiredCols,eeg_endCols),axis=1)
                    
                else:
                    # use for unequal epochs: fill steps with zeros 
                    # sort end values
                    idx_en22 = idx_en21 + dev_step.T 
                    i = len(idx_en22.T)
                    allArrays = []
                    for x in range(i):
                        myArray = avg[int(idx_en21[:,x]):int(idx_en22[:,x])]
                        allArrays.append(myArray)
                    eeg_endVals = allArrays
                        # add zeros to fill enable epochs of shorter length equal with epochs of adequate length
                    eeg_endVals = pd.DataFrame(eeg_endVals)
                    eeg_endVals.fillna(eeg_endVals.mean(),inplace=True)
                    eeg_endCols = eeg_endVals.values
                    
                    # merge the different sections of the epoch to form the epochs
                    epochs_dev = np.concatenate((eeg_startCols,eeg_desiredCols,eeg_endCols),axis=1)
                    # eeg_dev = np.concatenate((eeg_startCols,eeg_desiredCols,eeg_endCols),axis=1)
                
                # baseline correction
                prestim = epochs_dev[:,0:49]
                mean_prestim = np.mean(prestim,axis=1)
                mean_prestim = mean_prestim.reshape(len(mean_prestim),1)
                bc_dev = epochs_dev - mean_prestim
                
                # artefact rejection
                p2p = peaktopeak(bc_dev)
                result = np.where(p2p > clip)
                row = result[0]
                ar_dev = np.delete(bc_dev,(row),axis = 0)
                filtering = filters()
                ar_dev = filtering.butter_lowpass(ar_dev.T,lowpass,fs)
                ar_dev = ar_dev.T
                dif = ((len(bc_dev)-len(ar_dev))/len(bc_dev))*100
                if len(ar_dev) == len(bc_dev):
                    if dispIMG == True:
                        print("notice! epochs lost for dev tone:","{:.2%}".format((int(dif))/100))
                    else:
                        pass
                elif len(ar_dev) < len(bc_dev):
                    if dispIMG == True:
                        print("callback! epochs lost for dev tone:","{:.2%}".format((int(dif))/100))
                    else:
                        pass
                
                if ar_dev.size == 0:
                    epochLen = bc_dev.shape[1]
                    noEpochs = 300
                    ar_dev = np.full((noEpochs,epochLen),np.nan)
                    avg_dev = np.nanmean(ar_dev,axis=0)
                    print(chanNames,':',scanID," removed for deviant tones N1P3 analysis as all its epochs exceed the clip value of",clip)
                elif ar_dev.size != 0:
                    ar_dev = addNan(ar_dev)
                    avg_dev = np.nanmean(ar_dev,axis=0)
                avg_dev = avg_dev
                return avg_std,avg_dev,ar_std,ar_dev
            # algorithm function returns avg_std,avg_dev,ar_std,ar_dev
            out_final = []
            for i in range(len(channel_data.T)):
                out_final.append(algorithm(chanNames[i],scanID,trigger_channel,channel_data[:,i],period,stimTrig,clip,lowpass,fs,dispIMG))
            out_final = np.asarray(out_final,dtype="object").T
            out_final = out_final.transpose()
            return out_final

        N1P3 = N100P300_(eegData,period,trigger_channel,scanID,chanNames,stimTrig,clip,lowpass,fs,dispIMG)
        N1P3_Fz = N1P3[0]
        N1P3_Cz = N1P3[1]
        N1P3_Pz = N1P3[2]
        erp_latency = np.array(np.linspace(start=-100, stop=900, num=len(N1P3_Fz[0]),dtype=object),dtype=object)

        #   stack up ERPs
        std_Fz,dev_Fz = N1P3_Fz[0],N1P3_Fz[1]
        std_Cz,dev_Cz = N1P3_Cz[0],N1P3_Cz[1]
        std_Pz,dev_Pz = N1P3_Pz[0],N1P3_Pz[1]
        ERPs = np.vstack((std_Fz,dev_Fz,std_Cz,dev_Cz,std_Pz,dev_Pz))

        #   stack up epochs
        std_Fz,dev_Fz = N1P3_Fz[2],N1P3_Fz[3]
        std_Cz,dev_Cz = N1P3_Cz[2],N1P3_Cz[3]
        std_Pz,dev_Pz = N1P3_Pz[2],N1P3_Pz[3]
        epochs = np.stack((std_Fz,dev_Fz,std_Cz,dev_Cz,std_Pz,dev_Pz))
        return ERPs,epochs,erp_latency

    def N400(self,eegData,period,trigger_channel,scanID,chanNames,stimTrig,clip,lowpass,fs,dispIMG):
        """
          Inputs: trigger channels, bandpass filtered data for all channels, time period,
                  stimTrig, clip value
        """
        def addNan(x):
            noEpochs = 300
            lenRow = len(x)
            lenCol = len(x.T)
            remainRows = int(noEpochs-lenRow)
            nansValues = np.full((remainRows,lenCol),np.nan)
            return np.vstack((x,nansValues))
        
        def N400_(eegData,period,trigger_channel,scanID,chanNames,stimTrig,clip,lowpass,fs,dispIMG):
            chanNames,scanID,trigger_channel,channel_data,period,stimTrig,clip,lowpass,fs,dispIMG = chanNames,scanID,trigger_channel,eegData,period,stimTrig,clip,lowpass,fs,dispIMG
            def algorithm(chanNames,scanID,trigger_channel,channel_data,period,stimTrig,clip,lowpass,fs,dispIMG):
                trigger_data = rising_edge(trigger_channel)
                trigCol = trigger_data
                avg = channel_data 
                Ts = period
                # congruent word [4,7]: extract the time points where stimuli 1 exists
                con = stimTrig['con']
                con = con[0:2]
                con = (np.array([con])).T
                
                no_fours = np.count_nonzero(trigCol==4)
                no_sevens = np.count_nonzero(trigCol==7)
                no_cons = no_fours + no_sevens
                if dispIMG == True:
                    print("number of con word event codes:",no_cons)
                else:
                    pass
                
                result = np.where(trigCol == con[0])
                idx_30i = result[0]
                idx_30i = idx_30i.reshape(len(idx_30i),1)
                result = np.where(trigCol == con[1])
                idx_30ii = result[0]
                idx_30ii = idx_30ii.reshape(len(idx_30ii),1)
                idx_30 = np.vstack((idx_30i,idx_30ii))
                idx_31 = idx_30.reshape(1,len(idx_30))
                
                
                # sort target values
                desiredCols = trigCol[idx_30]
                desiredCols = desiredCols.reshape(len(desiredCols),1)
                
                # sort values before target
                idx_con = (idx_31 - 50)
                i = len(idx_con.T)
                allArrays = np.array([])
                for x in range(i):
                    myArray = trigCol[int(idx_con[:,x]):int(idx_31[:,x])]
                    allArrays = np.concatenate([allArrays,myArray])
                startVals = allArrays
                startCols = startVals.reshape(no_cons,50)
                
                # sort values immediately after target
                idx_en31 = idx_31 + 1
                postTargetCols = trigCol[idx_en31]
                postTargetCols = postTargetCols.T
                
                # sort end values
                    # ----------------------------------------------------------------------------
                    # determination of the number steps to reach the last point of each epoch
                    # the event codes are not evenly timed hence the need for steps determination
                a = trigCol[idx_30]
                a = a.reshape(len(a),1)
                
                b = Ts[idx_30]
                b = b.reshape(len(idx_30),1)
                
                c_i = float("%0.3f" % (Ts[int(len(Ts)-1)]))
                c = (np.array([0,c_i]))
                c = c.reshape(1,len(c))
                
                con_distr = np.concatenate((a, b),axis=1)
                con_distr = np.vstack((con_distr,c))
                
                con_diff = np.diff(con_distr[:,1],axis=0)
                con_step = ((con_diff/0.002)-1)
                con_step = np.where(con_step > 447, 447, con_step)
                
                # check if the number of steps are the same if yes, meaning the epochs have the same length
                result = np.all(con_step == con_step[0])
                if result:
                    # use for equal epoch steps
                    # sort end values
                    idx_en32 = idx_en31 + con_step.T
                    i = len(idx_en32.T)
                    allArrays = np.array([])
                    for x in range(i):
                        myArray = trigCol[int(idx_en31[:,x]):int(idx_en32[:,x])]
                        allArrays = np.concatenate([allArrays,myArray])
                    endVals = allArrays
                    endCols = endVals.reshape(no_cons,447)
                    
                    # merge the different sections of the epoch to form the epochs
                    trig_con = np.concatenate((startCols,desiredCols,endCols),axis=1)
                
                else:
                    # use when we have unequal epoch steps
                        # --------------------------------------------
                        # apply the step to get the index of the last point of the epoch
                    idx_en32 = idx_en31 + con_step.T 
                    i = len(idx_en32.T)
                    allArrays = []
                    for x in range(i):
                        myArray = trigCol[int(idx_en31[:,x]):int(idx_en32[:,x])]
                        allArrays.append(myArray)
                    endVals = allArrays
                        # add zeros to fill enable epochs of shorter length equal with epochs of adequate length
                    endVals = pd.DataFrame(endVals)
                    endVals.fillna(endVals.mean(),inplace=True)
                    endCols = endVals.values
                    
                    # merge the different sections of the epoch to form the epochs
                    trig_conpad = np.concatenate((startCols,desiredCols,endCols),axis=1)
                
                # ----------------------------------------------------------------------------------------------------------------
                # implement trig epoch creation to eeg data
                    # replace trigCol with avg 
                    # get start, desired, post and end col, then merge together
                    # remove data equal to non 1 or o triggers
                
                # sort target values
                eeg_desiredCols = avg[idx_30]
                eeg_desiredCols = eeg_desiredCols.reshape(len(eeg_desiredCols),1)
                
                # sort values before target
                i = len(idx_con.T)
                allArrays = np.array([])
                for x in range(i):
                    myArray = avg[int(idx_con[:,x]):int(idx_31[:,x])]
                    allArrays = np.concatenate([allArrays,myArray])
                eeg_startVals = allArrays
                eeg_startCols = eeg_startVals.reshape(no_cons,50)
                
                # sort values immediately after target
                eeg_postTargetCols = avg[idx_en31]
                eeg_postTargetCols = eeg_postTargetCols.T
                
                # check if the number of steps are the same if yes, meaning the epochs have the same length
                result = np.all(con_step == con_step[0])
                if result:
                    # use for equal epochs
                    # sort end values
                    idx_en32 = idx_en31 + con_step.T
                    i = len(idx_en32.T)
                    allArrays = np.array([])
                    for x in range(i):
                        myArray = avg[int(idx_en31[:,x]):int(idx_en32[:,x])]
                        allArrays = np.concatenate([allArrays,myArray])
                    endVals = allArrays
                    eeg_endCols = endVals.reshape(no_cons,447)
                    
                    # merge the different sections of the epoch to form the epochs
                    # eeg_std = np.concatenate((eeg_startCols,eeg_desiredCols,eeg_endCols),axis=1)
                    epochs_con = np.concatenate((eeg_startCols,eeg_desiredCols,eeg_endCols),axis=1)
                    
                else:
                    # use for unequal epochs
                    # sort end values
                    idx_en32 = idx_en31 + con_step.T 
                    i = len(idx_en32.T)
                    allArrays = []
                    for x in range(i):
                        myArray = avg[int(idx_en31[:,x]):int(idx_en32[:,x])]
                        allArrays.append(myArray)
                    eeg_endVals = allArrays
                        # add zeros to fill enable epochs of shorter length equal with epochs of adequate length
                    eeg_endVals = pd.DataFrame(eeg_endVals)
                    eeg_endVals.fillna(eeg_endVals.mean(),inplace=True)
                    eeg_endCols = eeg_endVals.values
                    
                    # merge the different sections of the epoch to form the epochs
                    epochs_con = np.concatenate((eeg_startCols,eeg_desiredCols,eeg_endCols),axis=1)
                
                
                # baseline correction
                prestim = epochs_con[:,0:49]
                mean_prestim = np.mean(prestim,axis=1)
                mean_prestim = mean_prestim.reshape(len(mean_prestim),1)
                bc_con = epochs_con - mean_prestim
                
                # artefact rejection
                p2p = peaktopeak(bc_con)
                result = np.where(p2p > clip)
                row = result[0]
                ar_con = np.delete(bc_con,(row),axis = 0)
                #filtering = filters()
                #ar_con = filtering.butter_lowpass(ar_con.T,lowpass,fs)
                #ar_con = ar_con.T
                dif = ((len(bc_con)-len(ar_con))/len(bc_con))*100
                if len(ar_con) == len(bc_con):
                    if dispIMG == True:
                        print("notice! epochs lost for con word:","{:.2%}".format((int(dif))/100))
                    else:
                        pass
                elif len(ar_con) < len(bc_con):
                    if dispIMG == True:
                        print("callback! epochs lost for con word:","{:.2%}".format((int(dif))/100))
                    else:
                        pass

                if ar_con.size == 0:
                    epochLen = bc_con.shape[1]
                    noEpochs = 300
                    ar_con = np.full((noEpochs,epochLen),np.nan)
                    avg_con = np.nanmean(ar_con,axis=0)
                    print(chanNames,':',scanID," removed for congruent words N4 analysis as all its epochs exceed the clip value of",clip)
                else:
                    # averaging
                    ar_con = addNan(ar_con)
                    avg_con = np.nanmean(ar_con,axis=0)
                avg_con = avg_con
                
                
                # %%
                # incongruent word [4,7]: extract the time points where stimuli 1 exists
                inc = stimTrig['inc']
                inc = inc[0:2]
                inc = (np.array([inc])).T
                
                no_fives = np.count_nonzero(trigCol==5)
                no_eights = np.count_nonzero(trigCol==8)
                no_incs = no_fives + no_eights
                if dispIMG == True:
                    print("number of inc word event codes:",no_incs)
                else:
                    pass
                
                result = np.where(trigCol == inc[0])
                idx_40i = result[0]
                idx_40i = idx_40i.reshape(len(idx_40i),1)
                result = np.where(trigCol == inc[1])
                idx_40ii = result[0]
                idx_40ii = idx_40ii.reshape(len(idx_40ii),1)
                idx_40 = np.vstack((idx_40i,idx_40ii))
                idx_41 = idx_40.reshape(1,len(idx_40))
                
                
                # sort target values
                desiredCols = trigCol[idx_40]
                desiredCols = desiredCols.reshape(len(desiredCols),1)
                
                # sort values before target
                idx_inc = (idx_41 - 50)
                i = len(idx_inc.T)
                allArrays = np.array([])
                for x in range(i):
                    myArray = trigCol[int(idx_inc[:,x]):int(idx_41[:,x])]
                    allArrays = np.concatenate([allArrays,myArray])
                startVals = allArrays
                startCols = startVals.reshape(no_incs,50)
                
                # sort values immediately after target
                idx_en41 = idx_41 + 1
                postTargetCols = trigCol[idx_en41]
                postTargetCols = postTargetCols.T
                
                # sort end values
                    # ----------------------------------------------------------------------------
                    # determination of the number steps to reach the last point of each epoch
                    # the event codes are not evenly timed hence the need for steps determination
                a = trigCol[idx_40]
                a = a.reshape(len(a),1)
                
                b = Ts[idx_40]
                b = b.reshape(len(idx_40),1)
                
                c_i = float("%0.3f" % (Ts[int(len(Ts)-1)]))
                c = (np.array([0,c_i]))
                c = c.reshape(1,len(c))
                
                inc_distr = np.concatenate((a, b),axis=1)
                inc_distr = np.vstack((inc_distr,c))
                
                inc_diff = np.diff(inc_distr[:,1],axis=0)
                inc_step = ((inc_diff/0.002)-1)
                inc_step = np.where(inc_step > 447, 447, inc_step)
                
                # check if the number of steps are the same if yes, meaning the epochs have the same length
                result = np.all(inc_step == inc_step[0])
                if result:
                    # use for equal epoch steps
                    # sort end values
                    idx_en42 = idx_en41 + inc_step.T
                    i = len(idx_en42.T)
                    allArrays = np.array([])
                    for x in range(i):
                        myArray = trigCol[int(idx_en41[:,x]):int(idx_en42[:,x])]
                        allArrays = np.concatenate([allArrays,myArray])
                    endVals = allArrays
                    endCols = endVals.reshape(no_inc,447)
                    
                    # merge the different sections of the epoch to form the epochs
                    trig_inc = np.concatenate((startCols,desiredCols,endCols),axis=1)
                
                else:
                    # use when we have unequal epoch steps
                        # --------------------------------------------
                        # apply the step to get the index of the last point of the epoch
                    idx_en42 = idx_en41 + inc_step.T 
                    i = len(idx_en42.T)
                    allArrays = []
                    for x in range(i):
                        myArray = trigCol[int(idx_en41[:,x]):int(idx_en42[:,x])]
                        allArrays.append(myArray)
                    endVals = allArrays
                        # add zeros to fill enable epochs of shorter length equal with epochs of adequate length
                    l = endVals
                    max_len = max([len(arr) for arr in l])
                    padded = np.array([np.lib.pad(arr, (0, max_len - len(arr)), 'constant', constant_values=0) for arr in l])
                        # appropriate end columns
                    endCols = padded
                    
                    # merge the different sections of the epoch to form the epochs
                    trig_incpad = np.concatenate((startCols,desiredCols,endCols),axis=1)
                
                
                # ----------------------------------------------------------------------------------------------------------------
                # implement trig epoch creation to eeg data
                    # replace trigCol with avg 
                    # get start, desired, post and end col, then merge together
                    # remove data equal to non 1 or o triggers
                
                # sort target values
                eeg_desiredCols = avg[idx_40]
                eeg_desiredCols = eeg_desiredCols.reshape(len(eeg_desiredCols),1)
                
                # sort values before target
                i = len(idx_inc.T)
                allArrays = np.array([])
                for x in range(i):
                    myArray = avg[int(idx_inc[:,x]):int(idx_41[:,x])]
                    allArrays = np.concatenate([allArrays,myArray])
                eeg_startVals = allArrays
                eeg_startCols = eeg_startVals.reshape(no_incs,50)
                
                # sort values immediately after target
                eeg_postTargetCols = avg[idx_en41]
                eeg_postTargetCols = eeg_postTargetCols.T
                
                # check if the number of steps are the same if yes, meaning the epochs have the same length
                result = np.all(inc_step == inc_step[0])
                if result:
                    # use for equal epochs
                    # sort end values
                    idx_en42 = idx_en41 + inc_step.T
                    i = len(idx_en42.T)
                    allArrays = np.array([])
                    for x in range(i):
                        myArray = avg[int(idx_en41[:,x]):int(idx_en42[:,x])]
                        allArrays = np.concatenate([allArrays,myArray])
                    endVals = allArrays
                    eeg_endCols = endVals.reshape(no_incs,447)
                    
                    # merge the different sections of the epoch to form the epochs
                    # eeg_std = np.concatenate((eeg_startCols,eeg_desiredCols,eeg_endCols),axis=1)
                    epochs_inc = np.concatenate((eeg_startCols,eeg_desiredCols,eeg_endCols),axis=1)
                    
                else:
                    # use for unequal epochs
                    # sort end values
                    idx_en42 = idx_en41 + inc_step.T 
                    i = len(idx_en42.T)
                    allArrays = []
                    for x in range(i):
                        myArray = avg[int(idx_en41[:,x]):int(idx_en42[:,x])]
                        allArrays.append(myArray)
                    eeg_endVals = allArrays
                        # add zeros to fill enable epochs of shorter length equal with epochs of adequate length
                    eeg_endVals = pd.DataFrame(eeg_endVals)
                    eeg_endVals.fillna(eeg_endVals.mean(),inplace=True)
                    eeg_endCols = eeg_endVals.values
                    
                    # merge the different sections of the epoch to form the epochs
                    epochs_inc = np.concatenate((eeg_startCols,eeg_desiredCols,eeg_endCols),axis=1)
                
                # baseline correction
                prestim = epochs_inc[:,0:49]
                mean_prestim = np.mean(prestim,axis=1)
                mean_prestim = mean_prestim.reshape(len(mean_prestim),1)
                bc_inc = epochs_inc - mean_prestim
                
                # artefact rejection
                p2p = peaktopeak(bc_inc)
                result = np.where(p2p > clip)
                row = result[0]
                ar_inc = np.delete(bc_inc,(row),axis = 0)
                #filtering = filters()
                #ar_inc = filtering.butter_lowpass(ar_inc.T,lowpass,fs)
                #ar_inc = ar_inc.T
                dif = ((len(bc_inc)-len(ar_inc))/len(bc_inc))*100
                if len(ar_inc) == len(bc_inc):
                    if dispIMG == True:
                        print("notice! epochs lost for inc word:","{:.2%}".format((int(dif))/100))
                    else:
                        pass
                elif len(ar_inc) < len(bc_inc):
                    if dispIMG == True:
                        print("callback! epochs lost for inc word:","{:.2%}".format((int(dif))/100))
                    else:
                        pass

                if ar_inc.size == 0:
                    epochLen = bc_inc.shape[1]
                    noEpochs = 300
                    ar_inc = np.full((noEpochs,epochLen),np.nan)
                    avg_inc = np.nanmean(ar_inc,axis=0)
                    print(chanNames,':',scanID," removed for incongruent words N4 analysis as all its epochs exceed the clip value of",clip)
                else:
                    # averaging
                    ar_inc = addNan(ar_inc)
                    avg_inc = np.nanmean(ar_inc,axis=0)
                avg_inc = avg_inc
                return avg_con,avg_inc,ar_con,ar_inc
            # algorithm function returns avg_con,avg_inc,ar_con,ar_inc   
            out_final = []
            for i in range(len(channel_data.T)):
                out_final.append(algorithm(chanNames[i],scanID,trigger_channel,channel_data[:,i],period,stimTrig,clip,lowpass,fs,dispIMG))
            out_final = np.asarray(out_final,dtype="object").T
            out_final = out_final.transpose()
            return out_final
        
        N4 = N400_(eegData,period,trigger_channel,scanID,chanNames,stimTrig,clip,lowpass,fs,dispIMG)
        N4_Fz = N4[0]
        N4_Cz = N4[1]
        N4_Pz = N4[2]
        erp_latency = np.array(np.linspace(start=-100, stop=900, num=len(N4_Fz[0]),dtype=object),dtype=object)

        #   stack up ERPs
        con_Fz,inc_Fz = N4_Fz[0],N4_Fz[1]
        con_Cz,inc_Cz = N4_Cz[0],N4_Cz[1]
        con_Pz,inc_Pz = N4_Pz[0],N4_Pz[1]
        ERPs = np.vstack((con_Fz,inc_Fz,con_Cz,inc_Cz,con_Pz,inc_Pz))

        #   stack up epochs
        con_Fz,inc_Fz = N4_Fz[2],N4_Fz[3]
        con_Cz,inc_Cz = N4_Cz[2],N4_Cz[3]
        con_Pz,inc_Pz = N4_Pz[2],N4_Pz[3]
        epochs = np.stack((con_Fz,inc_Fz,con_Cz,inc_Cz,con_Pz,inc_Pz))
        return ERPs,epochs,erp_latency

class visualizations:

    def PSD(self,data,fs,win,plot_title,xlim,figsize):
        """
        inputs: data = (samples,channels)
                fs = sampling rate
                plot_title = title of plot
                xlim = x-axis limits
        """
        fig,axis = plt.subplots(1,len(data.T),constrained_layout=True, sharey=True,figsize=(figsize[0],figsize[1]))
        fig.set_dpi(600)
        for i in range(len(data.T)):
            freqs,psd = signal.welch(data[:,i],fs,nperseg=win)
            axis[i].plot(freqs,psd)
            axis[i].set_title(plot_title[i])
            axis[i].set_xlim([xlim[0],xlim[1]])
            axis[i].set(xlabel='Frequency (Hz)', ylabel='Power Spectral Density (uV\u00b2/Hz)')
            axis[i].tick_params(axis='both', which='major', labelsize=8)
            axis[i].label_outer()

    def ERPs(self,array,erp_latency,titles,N1P3=False,N4=False):
        """
        inputs -    data: 3D array (channels,stimulusERP,length of ERP)
                    erp_latency: 1D array (length of ERP)
                    titles: a title per channel totaling three titles
        outputs -   3 subplots of ERPs for each channel
        note -      stimulusERP (standard tones ERP, deviant tones ERP, congruent words ERP, incongruent words ERP)
        """
        fig,axis = plt.subplots(1,int(len(array)/2),constrained_layout=True, sharey=True,figsize=(25,5))
        fig.set_dpi(1000)
        std_0,std_1,std_2 = np.nanstd(array[0]),np.nanstd(array[1]),np.nanstd(array[2])
        std_3,std_4,std_5 = np.nanstd(array[3]),np.nanstd(array[4]),np.nanstd(array[5])
        erp_latency = erp_latency.astype(np.float64)
        shade_degree,line_width = 0.05,5
        axis[0].plot(erp_latency,array[0],color='blue',linewidth=line_width)
        axis[0].plot(erp_latency,array[1],color='red',linewidth=line_width)
        axis[0].fill_between(erp_latency,array[0]-std_0,array[0]+std_0,color='blue',alpha=shade_degree)
        axis[0].fill_between(erp_latency,array[1]-std_1,array[1]+std_1,color='red',alpha=shade_degree)
        axis[0].set_title(titles[0])
        axis[0].set_ylabel('Amplitude (uV)')
        axis[0].set_xlabel('Time (ms)')
        axis[0].xaxis.set_major_locator(MultipleLocator(100))   
        axis[0].axvline(0, color='black', linestyle='--')
        axis[0].invert_yaxis()
        axis[0].set_ylim([7,-7])
        axis[1].plot(erp_latency,array[2],color='blue',linewidth=line_width)
        axis[1].plot(erp_latency,array[3],color='red',linewidth=line_width)
        axis[1].fill_between(erp_latency,array[2]-std_2,array[2]+std_2,color='blue',alpha=shade_degree)
        axis[1].fill_between(erp_latency,array[3]-std_3,array[3]+std_3,color='red',alpha=shade_degree)
        axis[1].set_title(titles[1])
        axis[1].set_xlabel('Time (ms)')
        axis[1].xaxis.set_major_locator(MultipleLocator(100))
        axis[1].axvline(0,color='black', linestyle='--')
        axis[1].invert_yaxis()
        axis[1].set_ylim([7,-7])
        axis[2].plot(erp_latency,array[4],color='blue',linewidth=line_width)
        axis[2].plot(erp_latency,array[5],color='red',linewidth=line_width)
        axis[2].fill_between(erp_latency,array[4]-std_4,array[4]+std_4,color='blue',alpha=shade_degree)
        axis[2].fill_between(erp_latency,array[5]-std_5,array[5]+std_5,color='red',alpha=shade_degree)
        axis[2].set_title(titles[2])
        axis[2].set_xlabel('Time (ms)')
        axis[2].xaxis.set_major_locator(MultipleLocator(100))
        axis[2].axvline(0,color='black', linestyle='--')
        axis[2].invert_yaxis()
        axis[2].set_ylim([7,-7])
        if N1P3:
            labels = ['standard tones','deviant tones']
        if N4:
            labels = ['congruent words','incongruent words']
        axis[2].legend(labels, loc = 'upper right')
        plt.show()


    def linePlots(self,x_data,y_data,plot_title,xlim,labels,figsize):
        fig,axis = plt.subplots(1,len(y_data.T),constrained_layout=True, sharey=True,figsize=(figsize[0],figsize[1]))
        fig.set_dpi(1000)
        for i in range(len(y_data.T)):
            axis[i].plot(x_data,y_data[:,i])
            axis[i].set_title(plot_title[i])
            axis[i].set_xlim([xlim[0],xlim[1]])
            axis[i].set(xlabel=labels[0], ylabel=labels[1])
            axis[i].tick_params(axis='both', which='major', labelsize=8)
            axis[i].label_outer()

    def spectrogram(self,time_s,data,fs,figsize,subTitles,xlim):
        if len(data.T)==4:
            eeg = data
            sr = fs
            WinLength = int(0.5*sr) 
            step = int(0.025*sr) 
            myparams = dict(nperseg = WinLength, noverlap = WinLength-step, scaling='density', return_onesided=True, mode='psd')
            f_1, nseg_1, Sxx_1 = signal.spectrogram(x = eeg[:,0], fs=sr, **myparams)
            f_2, nseg_2, Sxx_2 = signal.spectrogram(x = eeg[:,1], fs=sr, **myparams)
            f_3, nseg_3, Sxx_3 = signal.spectrogram(x = eeg[:,2], fs=sr, **myparams)
            f_4, nseg_4, Sxx_4 = signal.spectrogram(x = eeg[:,3], fs=sr, **myparams)

            fig, ax = plt.subplots(2,4, figsize = figsize, constrained_layout=True)
            fig.set_dpi(1000)
            #fig.suptitle(title)
            ax[0,0].plot(time_s, eeg[:,0], lw = 1, color='C0')
            ax[0,1].plot(time_s, eeg[:,1], lw = 1, color='C1')
            ax[0,2].plot(time_s, eeg[:,2], lw = 1, color='C2')
            ax[0,3].plot(time_s, eeg[:,3], lw = 1, color='C3')
            ax[0,0].set_ylabel('Amplitude ($\mu V$)')
            ax[0,1].set_ylabel('Amplitude ($\mu V$)')
            ax[0,2].set_ylabel('Amplitude ($\mu V$)')
            ax[0,3].set_ylabel('Amplitude ($\mu V$)')
            ax[0,0].set_xticks(np.arange(xlim[0],xlim[1],20))
            ax[0,1].set_xticks(np.arange(xlim[0],xlim[1],20))
            ax[0,2].set_xticks(np.arange(xlim[0],xlim[1],20))
            ax[0,3].set_xticks(np.arange(xlim[0],xlim[1],20))
            ax[0,0].set_title(subTitles[0])
            ax[0,1].set_title(subTitles[1])
            ax[0,2].set_title(subTitles[2])
            ax[0,3].set_title(subTitles[3])
            ax[1,0].set(xlabel='Time (s)', ylabel='Frequency (Hz)')
            ax[1,0].set_ylim(0,45)
            ax[1,1].set(xlabel='Time (s)', ylabel='Frequency (Hz)')
            ax[1,1].set_ylim(0,45)
            ax[1,2].set(xlabel='Time (s)', ylabel='Frequency (Hz)')
            ax[1,2].set_ylim(0,45)
            ax[1,3].set(xlabel='Time (s)', ylabel='Frequency (Hz)')
            ax[1,3].set_ylim(0,45)
            X1,X2,X3,X4 = nseg_1,nseg_2,nseg_3,nseg_4
            Y1,Y2,Y3,Y4 = f_1,f_2,f_3,f_4
            Z1,Z2,Z3,Z4 = Sxx_1,Sxx_2,Sxx_3,Sxx_4
            levels = 45
            spectrum = ax[1,0].contourf(X1,Y1,Z1,levels, cmap='jet')
            spectrum = ax[1,1].contourf(X2,Y2,Z2,levels, cmap='jet')
            spectrum = ax[1,2].contourf(X3,Y3,Z3,levels, cmap='jet')
            spectrum = ax[1,3].contourf(X4,Y4,Z4,levels, cmap='jet')
            ax[1,0].set_xticks(np.arange(xlim[0],xlim[1],20))
            ax[1,1].set_xticks(np.arange(xlim[0],xlim[1],20))
            ax[1,2].set_xticks(np.arange(xlim[0],xlim[1],20))
            ax[1,3].set_xticks(np.arange(xlim[0],xlim[1],20))
            cbar = plt.colorbar(spectrum)#, boundaries=np.linspace(0,1,5))
            cbar.ax.set_ylabel('Amplitude (dB)', rotation=90)
        if len(data.T)==3:
            eeg = data
            sr = fs
            WinLength = int(0.5*sr) 
            step = int(0.025*sr) 
            myparams = dict(nperseg = WinLength, noverlap = WinLength-step, scaling='density', return_onesided=True, mode='psd')
            f_1, nseg_1, Sxx_1 = signal.spectrogram(x = eeg[:,0], fs=sr, **myparams)
            f_2, nseg_2, Sxx_2 = signal.spectrogram(x = eeg[:,1], fs=sr, **myparams)
            f_3, nseg_3, Sxx_3 = signal.spectrogram(x = eeg[:,2], fs=sr, **myparams)

            fig, ax = plt.subplots(2,3, figsize = figsize, constrained_layout=True)
            #fig.suptitle(title)
            ax[0,0].plot(time_s, eeg[:,0], lw = 1, color='C0')
            ax[0,1].plot(time_s, eeg[:,1], lw = 1, color='C1')
            ax[0,2].plot(time_s, eeg[:,2], lw = 1, color='C2')
            ax[0,0].set_ylabel('Amplitude ($\mu V$)')
            ax[0,1].set_ylabel('Amplitude ($\mu V$)')
            ax[0,2].set_ylabel('Amplitude ($\mu V$)')
            ax[0,0].set_xticks(np.arange(xlim[0],xlim[1],20))
            ax[0,1].set_xticks(np.arange(xlim[0],xlim[1],20))
            ax[0,2].set_xticks(np.arange(xlim[0],xlim[1],20))
            ax[0,0].set_title(subTitles[0])
            ax[0,1].set_title(subTitles[1])
            ax[0,2].set_title(subTitles[2])
            ax[1,0].set(xlabel='Time (s)', ylabel='Frequency (Hz)')
            ax[1,0].set_ylim(0,45)
            ax[1,1].set(xlabel='Time (s)', ylabel='Frequency (Hz)')
            ax[1,1].set_ylim(0,45)
            ax[1,2].set(xlabel='Time (s)', ylabel='Frequency (Hz)')
            ax[1,2].set_ylim(0,45)
            X1,X2,X3 = nseg_1,nseg_2,nseg_3
            Y1,Y2,Y3 = f_1,f_2,f_3
            Z1,Z2,Z3 = Sxx_1,Sxx_2,Sxx_3
            levels = 45
            spectrum = ax[1,0].contourf(X1,Y1,Z1,levels, cmap='jet')
            spectrum = ax[1,1].contourf(X2,Y2,Z2,levels, cmap='jet')
            spectrum = ax[1,2].contourf(X3,Y3,Z3,levels, cmap='jet')
            ax[1,0].set_xticks(np.arange(xlim[0],xlim[1],20))
            ax[1,1].set_xticks(np.arange(xlim[0],xlim[1],20))
            ax[1,2].set_xticks(np.arange(xlim[0],xlim[1],20))
            cbar = plt.colorbar(spectrum)#, boundaries=np.linspace(0,1,5))
            cbar.ax.set_ylabel('Amplitude (dB)', rotation=90)
             
def rolling_window(data_array,timing_array,window_size,step_size):
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

def slidingWindow(X, window_length, stride1):
    shape = (X.shape[0] - window_length + 1, window_length)
    strides = (X.strides[0],) + X.strides
    rolled = np.lib.stride_tricks.as_strided(X, shape=shape, strides=strides)
    return rolled[np.arange(0,shape[0],stride1)]

def slidingWindow_2D(data_2D,timing,window_size,step):
    """
    Inputs:
    1. data_2D - 2D numpy array (d0=samples, d1=channels) of data
    2. timing_array - 1D numpy array (d0 = samples) of timing data
    3. len(data_array) == len(timing_array)
    4. window_size - number of samples to use in each window in seconds e.g. 1 is 1 second
    5. step_size - the step size in seconds e.g.0.5 is 0.5 seconds 

    Outputs:    
    1. data_windows - 3D numpy array (d0=channels, d0=windows, d1=window size) of data
    """

    def params(data_1D,timing_array,window_size,step_size):
        idx_winsize = np.where(timing_array == window_size)[0][0]
        idx_stepsize = np.where(timing_array == step_size)[0][0]
        frame_len, hop_len = idx_winsize,idx_stepsize
        frames = librosa.util.frame(data_1D, frame_length=frame_len, hop_length=hop_len)
        windowed_frames = (np.hanning(frame_len).reshape(-1, 1)*frames).T
        return windowed_frames
    out_final = []
    for i in range(len(data_2D.T)):
        out_final.append(params(data_2D[:,i],timing,window_size,step))
    out_final = np.asarray(out_final).T
    out_final = out_final.transpose()
    return out_final

def normData_analysis(data,rem_outliers):

    if rem_outliers==0:
        cleaned_data = data

        def n100_amp(cleaned_data):
            n100_amp_max = np.amax(cleaned_data)
            n100_amp_min = np.amin(np.array(list(filter(lambda a: a != 0, cleaned_data))))
            n100_amp_best = n100_amp_max
            n100_amp_mean = np.mean(cleaned_data)
            n100_amp_std = np.std(cleaned_data)
            return n100_amp_mean,n100_amp_best,n100_amp_max,n100_amp_min,n100_amp_std

        def n100_lat(cleaned_data):
            n100_lat_max = np.amax(cleaned_data)
            n100_lat_min = np.amin(np.array(list(filter(lambda a: a != 0, cleaned_data))))
            n100_lat_best = n100_lat_min
            n100_lat_mean = np.mean(cleaned_data)
            n100_lat_std = np.std(cleaned_data)
            return n100_lat_mean,n100_lat_best,n100_lat_max,n100_lat_min,n100_lat_std

        def p300_amp(cleaned_data):
            p300_amp_max = np.amax(cleaned_data)
            p300_amp_min = np.amin(np.array(list(filter(lambda a: a != 0, cleaned_data))))
            p300_amp_best = p300_amp_max
            p300_amp_mean = np.mean(cleaned_data)
            p300_amp_std = np.std(cleaned_data)
            return p300_amp_mean,p300_amp_best,p300_amp_max,p300_amp_min,p300_amp_std

        def p300_lat(cleaned_data):
            p300_lat_max = np.amax(cleaned_data)
            p300_lat_min = np.amin(np.array(list(filter(lambda a: a != 0, cleaned_data))))
            p300_lat_best = p300_lat_min
            p300_lat_mean = np.mean(cleaned_data)
            p300_lat_std = np.std(cleaned_data)
            return p300_lat_mean,p300_lat_best,p300_lat_max,p300_lat_min,p300_lat_std
        
        def n400_amp(cleaned_data):
            n400_amp_max = np.amax(cleaned_data)
            n400_amp_min = np.amin(np.array(list(filter(lambda a: a != 0, cleaned_data))))
            n400_amp_best = n400_amp_max
            n400_amp_mean = np.mean(cleaned_data)
            n400_amp_std = np.std(cleaned_data)
            return n400_amp_mean,n400_amp_best,n400_amp_max,n400_amp_min,n400_amp_std

        def n400_lat(cleaned_data):
            n400_lat_max = np.amax(cleaned_data)
            n400_lat_min = np.amin(np.array(list(filter(lambda a: a != 0, cleaned_data))))
            n400_lat_best = n400_lat_min
            n400_lat_mean = np.mean(cleaned_data)
            n400_lat_std = np.std(cleaned_data)
            return n400_lat_mean,n400_lat_best,n400_lat_max,n400_lat_min,n400_lat_std 

    if rem_outliers==1:
        #remove outliers
        data = data.sort_values(axis=0, ascending=True)
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        low = q1 - 1.5 * iqr
        high = q3 + 1.5 * iqr
        cleaned_data = data.loc[(data > low) & (data < high)]
        cleaned_data = cleaned_data.to_numpy()

        def n100_amp(cleaned_data):
            n100_amp_max = np.amax(cleaned_data)
            n100_amp_min = np.amin(np.array(list(filter(lambda a: a != 0, cleaned_data))))
            n100_amp_best = n100_amp_max
            n100_amp_mean = np.mean(cleaned_data)
            n100_amp_std = np.std(cleaned_data)
            return n100_amp_mean,n100_amp_best,n100_amp_max,n100_amp_min,n100_amp_std

        def n100_lat(cleaned_data):
            n100_lat_max = np.amax(cleaned_data)
            n100_lat_min = np.amin(np.array(list(filter(lambda a: a != 0, cleaned_data))))
            n100_lat_best = n100_lat_min
            n100_lat_mean = np.mean(cleaned_data)
            n100_lat_std = np.std(cleaned_data)
            return n100_lat_mean,n100_lat_best,n100_lat_max,n100_lat_min,n100_lat_std

        def p300_amp(cleaned_data):
            p300_amp_max = np.amax(cleaned_data)
            p300_amp_min = np.amin(np.array(list(filter(lambda a: a != 0, cleaned_data))))
            p300_amp_best = p300_amp_max
            p300_amp_mean = np.mean(cleaned_data)
            p300_amp_std = np.std(cleaned_data)
            return p300_amp_mean,p300_amp_best,p300_amp_max,p300_amp_min,p300_amp_std

        def p300_lat(cleaned_data):
            p300_lat_max = np.amax(cleaned_data)
            p300_lat_min = np.amin(np.array(list(filter(lambda a: a != 0, cleaned_data))))
            p300_lat_best = p300_lat_min
            p300_lat_mean = np.mean(cleaned_data)
            p300_lat_std = np.std(cleaned_data)
            return p300_lat_mean,p300_lat_best,p300_lat_max,p300_lat_min,p300_lat_std
        
        def n400_amp(cleaned_data):
            n400_amp_max = np.amax(cleaned_data)
            n400_amp_min = np.amin(np.array(list(filter(lambda a: a != 0, cleaned_data))))
            n400_amp_best = n400_amp_max
            n400_amp_mean = np.mean(cleaned_data)
            n400_amp_std = np.std(cleaned_data)
            return n400_amp_mean,n400_amp_best,n400_amp_max,n400_amp_min,n400_amp_std

        def n400_lat(cleaned_data):
            n400_lat_max = np.amax(cleaned_data)
            n400_lat_min = np.amin(np.array(list(filter(lambda a: a != 0, cleaned_data))))
            n400_lat_best = n400_lat_min
            n400_lat_mean = np.mean(cleaned_data)
            n400_lat_std = np.std(cleaned_data)
            return n400_lat_mean,n400_lat_best,n400_lat_max,n400_lat_min,n400_lat_std 

    return cleaned_data,n100_amp,n100_lat,n400_amp,n400_lat,p300_amp,p300_lat

def ebs(parameter,mean,max,min):
    if parameter=='amplitude':
        if mean > max:
            score = 100
        if mean < min:
            score = 0
        if (mean>min and mean<max):
            best = max
            score = (round((1 - abs((mean-best)/(max-min))).item(),2))*100
    if parameter=='latency':
        if mean > max:
            score = 0
        if mean < min:
            score = 100
        if (mean>min and mean<max):
            best = min
            score = (round((1 - abs((mean-best)/(max-min))).item(),2))*100
    return score

def ICA(input,fs):
    """
    Inputs:  input: 2D array of EEG data (samples x channels)
                fs: sampling frequency
    Outputs: restored signal (samples x channels)
    """

    def icaHighpass(data,cutoff,fs):
        def params_fnc(data,cutoff,fs,order=4):
            nyq = 0.5 * fs
            normal_cutoff = cutoff / nyq
            b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
            y = signal.filtfilt(b, a, data)
            return y
        filterEEG = []
        for i in range(len(data.T)):
            filterEEG.append(params_fnc(data.T[i],cutoff,fs))
        filterEEG = np.array(filterEEG).T
        return filterEEG

    def confidenceInterval(samples):
    #   At 95% significance level, tN -1 = 2.201
        means = np.mean(samples)
        std_dev = np.std(samples)
        standard_error = std_dev/np.sqrt(len(samples))
        lower_95_perc_bound = means - 2.201*standard_error
        upper_95_perc_bound = means + 2.201*standard_error
        return upper_95_perc_bound

    def setZeros(data,index):
        def params(data):
            return np.zeros(len(data))
        zeros = []
        for i in range(len(index)):
            zeros.append(params(data.T[index[i]]))
        zeros = np.array(zeros)
        return zeros

    def sampEntropy(data):
        def params(input):
            import antropy as ant
            return ant.sample_entropy(input)
        sampEn = []
        for i in range(len(data.T)):
            sampEn.append(params(data[:,i]))
        sampEn = np.array(sampEn)
        # replace inf with 0
        sampEn[np.isinf(sampEn)] = 0
        return sampEn


    hpEEG = icaHighpass(input,1,fs)
    ica_ = FastICA(n_components=len(input.T), random_state=0, tol=0.0001)
    comps = ica_.fit_transform(hpEEG)
    comps_kurtosis = kurtosis(comps)
    comps_skew = skew(comps)
    comps_sampEN = sampEntropy(comps)

    #   Computing CI on to set threshold
    threshold_kurt = confidenceInterval(comps_kurtosis)
    threshold_skew = confidenceInterval(comps_skew)
    threshold_sampEN = confidenceInterval(comps_sampEN)

    "compare threshold with extracted parameter values"
    #   Extract epochs
    bool_ArtfCompsKurt = [comps_kurtosis>threshold_kurt]
    idx_ArtfCompsKurt = np.asarray(np.where(bool_ArtfCompsKurt[0]==True))
    bool_ArtfCompsSkew = [comps_skew>threshold_skew]
    idx_ArtfCompsSkew = np.asarray(np.where(bool_ArtfCompsSkew[0]==True))
    bool_ArtfCompsSampEN = [comps_sampEN>threshold_sampEN]
    idx_ArtfCompsSampEN = np.asarray(np.where(bool_ArtfCompsSampEN[0]==True))

    #   Merge index of components detected as artifacts by kurtosis, skewness, and sample entropy
    idx_artf_comps = np.concatenate((idx_ArtfCompsKurt,idx_ArtfCompsSkew,idx_ArtfCompsSampEN),axis=1)
    idx_artf_comps = np.unique(idx_artf_comps)

    "Component identified as artifact is converted to arrays of zeros"
    rejected_comps = setZeros(comps,idx_artf_comps)


    "Return zero-ed ICs into the original windows per ICs"
    for i in range(len(idx_artf_comps)):
        idx_rejected_comps = np.arange(len(rejected_comps))
        comps.T[idx_artf_comps[i]] = rejected_comps[idx_rejected_comps[i]]


    "Recover clean signal from clean ICs"
    restored = ica_.inverse_transform(comps)
    return restored

def pipeline(filenames,deviceVersion,path,sfreq,line,highPass,lowPass,erp_data_path,erp_tones_filename,erp_words_filename,stimTriggers=False,clip=False,channel_names=False,label=False,img_name=False,
                destination_dir=False,EEG_SingleSubject=False,EEG_MultipleSubjects=False,EEG_GrandAverages=False,ERPs_SingleSubject=False,ERPs_MultipleSubjects=False,
                ERPs_GrandAverages=False,Epochs_SingleSubject=False,Epochs_MultipleSubjects=False,erp_plots=False):
    '''

    Inputs:     default arguments are for EEG processing 
                False arguments are for ERP processing
                (default) filename(s): list of filenames to be averaged
                (default) deviceVersion: '1.0' or '1.1'
                (default) path: path to the folder containing the data
                (default) sfreq: sampling frequency
                (default) line: line noise frequency
                (default) highPass: high pass filter frequency
                (default) lowPass: low pass filter frequency

    Outputs:    single subject bandpassed EEG data, multiple subjects bandpassed EEG data, mean of multiple subjects bandpassed EEG data, 
                single subject ERP data, multiple subjects ERP data, mean of multiple subjects ERP data
    '''
    def fnc_1(device_version,scan_ID,local_path,fs,line_,lowcut,highcut):
        print("Processed file: ",scan_ID)
        device = importFile.neurocatch()
        fileObjects = device.init(device_version,scan_ID,local_path,dispIMG=False)
        rawEEG = fileObjects[0]
        rawEOG = fileObjects[1]
        time = fileObjects[3]
        trigOutput = fileObjects[4]
        filtering = filters()
        adaptiveFilterOutput = filtering.adaptive(rawEEG,rawEOG)
        notchFilterOutput = filtering.notch(adaptiveFilterOutput,line_,fs)
        bandPassFilterOutput = filtering.butterBandPass(notchFilterOutput,lowcut,highcut,fs)
        bandPassFilterOutput = bandPassFilterOutput
        #  check if the columns in arrays have equal lengths

        return bandPassFilterOutput
    def fnc_2(device_version,scan_ID,local_path,fs,line_,lowcut,highcut,stimTrig,clip_,chanNames):
        print("Processed file: ",scan_ID)
        device = importFile.neurocatch()
        fileObjects = device.init(device_version,scan_ID,local_path,dispIMG=False)
        rawEEG = fileObjects[0]
        rawEOG = fileObjects[1]
        time = fileObjects[3]
        trigOutput = fileObjects[4]
        filtering = filters()
        adaptiveFilterOutput = filtering.adaptive(rawEEG,rawEOG)
        notchFilterOutput = filtering.notch(adaptiveFilterOutput,line_,fs)
        bandPassFilterOutput = filtering.butterBandPass(notchFilterOutput,lowcut,highcut,fs)
        bandPassFilterOutput = bandPassFilterOutput
        erps = erpExtraction()
        N1P3 = erps.N100P300(bandPassFilterOutput,time,trigOutput,scan_ID,chanNames,stimTrig,clip_,highcut,fs,dispIMG=False)
        N4 = erps.N400(bandPassFilterOutput,time,trigOutput,scan_ID,chanNames,stimTrig,clip_,highcut,fs,dispIMG=False)
        erp_latency = N1P3[2]
        return N1P3,N4,erp_latency
    if EEG_SingleSubject:
        chans = fnc_1(deviceVersion,filenames,path,sfreq,line,highPass,lowPass)
    if EEG_MultipleSubjects:
        chans = []
        for i in range(len(filenames)):
            chans.append(fnc_1(deviceVersion,filenames[i],path,sfreq,line,highPass,lowPass))
        chans = np.array(chans)
    if EEG_GrandAverages:
        chans = []
        for i in range(len(filenames)):
            chans.append(fnc_1(deviceVersion,filenames[i],path,sfreq,line,highPass,lowPass))
        chans = np.array(chans)
        chans = np.mean(chans,axis=0)
    if ERPs_SingleSubject:
        tuple = fnc_2(deviceVersion,filenames,path,sfreq,line,highPass,lowPass,stimTriggers,clip,channel_names)
        latency = tuple[2]
        std_dev  = tuple[0][0]
        con_inc = tuple[1][0]
        chans = [std_dev,con_inc,latency]
        if erp_plots:
            plotter = visualizations()
            args_1 = {'titles':cfg.channelNames,'N1P3':True}
            args_2 = {'titles':cfg.channelNames,'N4':True}
            plotter.ERPs(std_dev,latency,**args_1)
            plotter.ERPs(con_inc,latency,**args_2) 
    if Epochs_SingleSubject:
        tuple = fnc_2(deviceVersion,filenames,path,sfreq,line,highPass,lowPass,stimTriggers,clip,channel_names)
        std_dev  = tuple[0][1]
        con_inc = tuple[1][1]
        chans = [std_dev,con_inc]
    if ERPs_MultipleSubjects:
        std_dev = []
        con_inc = []
        latency = []
        for i in range(len(filenames)):
            tuple = fnc_2(deviceVersion,filenames[i],path,sfreq,line,highPass,lowPass,stimTriggers,clip,channel_names)
            latency.append(tuple[2])
            std_dev.append(tuple[0][0])
            con_inc.append(tuple[1][0])
        std_dev = np.array(std_dev)
        con_inc = np.array(con_inc)
        latency = np.array(latency)
        # std_dev = [std_Fz,dev_Fz,std_Cz,dev_Cz,std_Pz,dev_Pz] # (single subjects. 6 rows, erp length)
        # con_inc = [con_Fz,inc_Fz,con_Cz,inc_Cz,con_Pz,inc_Pz]
        # if any of the rows have NaNs, remove the whole array

        chans = [std_dev,con_inc,latency]
    if Epochs_MultipleSubjects:
        std_dev = []
        con_inc = []
        for i in range(len(filenames)):
            tuple = fnc_2(deviceVersion,filenames[i],path,sfreq,line,highPass,lowPass,stimTriggers,clip,channel_names)
            std_dev.append(tuple[0][1])
            con_inc.append(tuple[1][1])
        std_dev = np.array(std_dev)
        con_inc = np.array(con_inc)
        chans = [std_dev,con_inc]
    if ERPs_GrandAverages:
        std_dev = []
        con_inc = []
        latency = []
        for i in range(len(filenames)):
            tuple = fnc_2(deviceVersion,filenames[i],path,sfreq,line,highPass,lowPass,stimTriggers,clip,channel_names)
            latency.append(tuple[2])
            std_dev.append(tuple[0][0])
            con_inc.append(tuple[1][0])
        std_dev = np.array(std_dev)
        con_inc = np.array(con_inc)
        latency = np.array(latency)
        print("std_dev shape: ",std_dev.shape)
        print("con_inc shape: ",con_inc.shape)
        #dir_path = '/Users/joshuaighalo/Documents/codespace/eegDementia/repo/'
        #np.save(erp_data_path + 'subjectsTonesERPs.npy', std_dev)
        #np.save(dir_path + 'subjectsWordsERPs.npy', con_inc)
        np.save(erp_data_path + erp_tones_filename, std_dev)
        np.save(erp_data_path + erp_words_filename, con_inc)
        std_dev = np.nanmean(std_dev,axis=0)
        con_inc = np.nanmean(con_inc,axis=0)
        latency = np.nanmean(latency,axis=0)
        chans = [std_dev,con_inc,latency]
        # chans = [(std_Fz,dev_Fz,std_Cz,dev_Cz,std_Pz,dev_Pz),(con_Fz,inc_Fz,con_Cz,inc_Cz,con_Pz,inc_Pz),latency]
        if erp_plots:
            plotter = visualizations()
            args_1 = {'titles':cfg.channelNames,'N1P3':True}
            args_2 = {'titles':cfg.channelNames,'N4':True}
            plotter.ERPs(std_dev,latency,**args_1)
            plotter.ERPs(con_inc,latency,**args_2)
    return chans

def signal_quality_index(eeg_1D,eeg_timePeriod,filename,dispIMG):
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

    A semi-simulated EEG/EOG dataset for the comparison of EOG artifact rejection techniques
    """

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

def psd(data,fs,data_1D=False,data_2D=False,data_3D=False):
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
        psd, freqs = psd_array_multitaper(dataIN, fs, adaptive=True,
                                            normalization='full', verbose=0)
        return freqs,psd
    def params_2D(dataIN,fs):
        freqs,psd = [],[]
        for i in range(len(dataIN.T)):
            f,p = params_1D(dataIN[:,i],fs)
            freqs.append(f)
            psd.append(p)
        psd = np.array(psd)
        freqs = np.array(freqs)
        return freqs,psd

    if data_1D:
        freqs,psd = params_1D(data,fs)
    if data_2D:
        freqs,psd = params_2D(data,fs)
    if data_3D:
        freqs,psd = [],[]
        for i in range(len(data)):
            f,p = params_2D(data[i,:,:],fs)
            freqs.append(f)
            psd.append(p)
        freqs = np.array(freqs)
        psd = np.array(psd)
    return freqs,psd

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
