"""
cleanEEG.py script takes a suggested clean eeg and processes it to its clean state
The Raw EEG (containing data from channels Fz, Cz & Pz) is from the bruyere dataset collected with neurocatch 1.1
The script only delivers channels Cz which is the cleanest amongst all channels
"""
from fn_cfg import *
import params as cfg


def DWT(data,time_array):
    #   Probability Mapping Based Artifact Detection and Wavelet Denoising based 
    #   Artifact Removal from Scalp EEG for BCI Applications
    #  Perform DWT on the data
    #   Input: data - EEG data: 1D array (samples x channel)
    #   Output: new signal: (samples x number of wavelets)
    #           signal_global - new signal extracted after global threshold 
    #           signal_std - new signal extracted after std threshold 
    #   Reference:  choice of number of levels to threshold gotten from "Comparative Study of Wavelet-Based Unsupervised 
    #               Ocular Artifact Removal Techniques for Single-Channel EEG Data"
      
    wavelets = ['bior4.4'] 
    def dwt_only(data,wavelet):
        def dwt(data,wavelet):
            coeffs = wavedec(data,wavelet,level=10)
            return np.array(coeffs,dtype=object).T

        def global_threshold(data,coeffs):
            def coeffs_approx(data,coeffs):
                return (np.median(abs(coeffs[0]))/0.6745)*(np.sqrt(2*np.log(len(data))))
            def coeffs_detail(data,coeffs):
                return (np.median(abs(coeffs[1]))/0.6745)*(np.sqrt(2*np.log(len(data))))
            arr_approx = coeffs_approx(data,coeffs)
            arr_detail = coeffs_detail(data,coeffs)
            return np.vstack((arr_approx,arr_detail))

        def apply_threshold(coeffs,threshold):
            def apply_threshold_approx(coeffs,threshold):
                coeffs[0][abs(coeffs[0])>threshold[1]] = 0
                coeffs_approx = coeffs[0]
                return coeffs_approx
            def apply_threshold_detail(coeffs,threshold):
                coeffs = coeffs[1:len(coeffs)]
                coeffs[0][abs(coeffs[0])>threshold[1]] = 0
                coeffs[1][abs(coeffs[1])>threshold[1]] = 0
                coeffs[2][abs(coeffs[2])>threshold[1]] = 0  # level 8
                coeffs[3][abs(coeffs[3])>threshold[1]] = 0  # level 7
                coeffs[4][abs(coeffs[4])>threshold[1]] = 0  # level 6
                coeffs[5][abs(coeffs[5])>threshold[1]] = 0  # level 5
                coeffs[6][abs(coeffs[6])>threshold[1]] = 0  # level 4
                coeffs[7][abs(coeffs[7])>threshold[1]] = 0  # level 3
                coeffs[8][abs(coeffs[8])>threshold[1]] = 0
                coeffs[9][abs(coeffs[9])>threshold[1]] = 0
                return coeffs
            arr_approx = apply_threshold_approx(coeffs,threshold)
            arr_detail = apply_threshold_detail(coeffs,threshold)
            arr_detail = list(np.array(arr_detail).T)
            arr_approx = arr_approx
            coefs = arr_detail
            (coefs).insert(0,arr_approx)
            return coefs

        def inv_dwt(coeffs,wavelet):
            def inverse_dwt(coeffs,wavelet):
                return waverec(coeffs,wavelet)
            arr = (inverse_dwt(list(np.array(coeffs,dtype=object)),wavelet))
            return  (np.array(arr).T)[:-1]

        coeffs = dwt(data,wavelet)
        threshold_global = global_threshold(data,coeffs)
        coeffs_global = apply_threshold(coeffs,threshold_global)
        signal_global = inv_dwt(coeffs_global,wavelet)
        return signal_global

    newEEG_global = []
    for i in range(len(wavelets)):
        newEEG_global.append((dwt_only(data,wavelets[i])))
    newEEG_global = np.array(newEEG_global).T
    if len(newEEG_global) != len(time_array):
        if len(newEEG_global) > len(time_array):
            diff = len(newEEG_global) - len(time_array)
            newEEG_global = newEEG_global[:-diff,:]
        elif len(newEEG_global) < len(time_array):
            diff = len(time_array) - len(newEEG_global)
            num_zeros = np.zeros((diff,len(newEEG_global[1])))
            newEEG_global = np.append(newEEG_global,num_zeros,axis=0)
    else:
        newEEG_global = newEEG_global
    return newEEG_global

def pipeline_single_NC(device_version,scan_ID,local_path,line,fs,lowcut,highcut):
    print("scan name: ",scan_ID)
    device = importFile.neurocatch()
    fileObjects = device.init(device_version,scan_ID,local_path)
    rawEEG = fileObjects[0]
    rawEOG = fileObjects[1]
    time = fileObjects[3]
    filtering = filters()
    dwt_Fz = DWT(rawEEG[:,0],time)
    dwt_Cz = DWT(rawEEG[:,1],time)
    dwt_Pz = DWT(rawEEG[:,2],time)
    dwt_EEG = np.hstack((dwt_Fz,dwt_Cz,dwt_Pz))
    adaptiveFilterOutput = filtering.adaptive(dwt_EEG,rawEOG)
    #adaptiveFilterOutput = filtering.adaptive(adaptiveFilterOutput,rawEOG)
    notchFilterOutput = filtering.notch(adaptiveFilterOutput,line,fs)
    bandPassFilterOutput = filtering.butterBandPass(notchFilterOutput,lowcut,highcut,fs)
    return bandPassFilterOutput


cleanEEG = pipeline_single_NC(device_version=1.1,scan_ID='0_1_12072018_1206',
                            local_path='/Users/joshuaighalo/Downloads/EEG_Datasets/bruyere',
                            line=60,fs=500,lowcut=0.1,highcut=10)
cleanEEG = cleanEEG[:,1]

plt.plot(cleanEEG)