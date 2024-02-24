"""
ICA using FastICA algorithm without sliding window
"""

from fn_cfg import *
import params as cfg
from scipy.stats import skew
from sklearn.decomposition import FastICA
import antropy as ant

"Laurel Place Dataset"
localPath = '/Users/joshuaighalo/Downloads/brainNet_datasets/laurel_place/cleaned_dataset'
img_save_path = '/Users/joshuaighalo/Documents/BrainNet/Projects/Workspace/results/laurel place/images/single_averages/2.0'
filename = '0002_1_12122019_1219'
version = 1.0
image_name = "Dementia Scan"
cutoff = 10
device = importFile.neurocatch()
fileObjects = device.init(version,filename,localPath,dispIMG=False)
rawEEG = fileObjects[0]
rawEOG = fileObjects[1]
rawEEGEOG = fileObjects[2]
time = fileObjects[3]



#   Implement high pass filter @ 1Hz
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

def sample_entropy(data):
    def params(data):
        return ant.sample_entropy(data)
    sample_entropy = []
    for i in range(len(data)):
        sample_entropy.append(params(data[i]))
    return sample_entropy

def confidenceInterval(samples):
#   At 95% significance level, tN -1 = 2.201
    means = np.mean(samples)
    std_dev = np.std(samples)
    standard_error = std_dev/np.sqrt(len(samples))
    lower_95_perc_bound = means - 2.201*standard_error
    upper_95_perc_bound = means + 2.201*standard_error
    return upper_95_perc_bound

def dwt(data,wavelet):
    # A5,D5,D4,D3,D2,D1 = 0-4Hz,8-16Hz,16-32Hz,32-64Hz,64-128Hz
    # A5 is eliminated: OAs are associated with freq ranged < 5Hz
    # D2,D1 are eliminated: MAs are associated with freq ranged > 32Hz
    coeffs = wavedec(data,wavelet,level=5)
    coeffs = np.array(coeffs,dtype=object)
    A5,D5,D4,D3,D2,D1 = coeffs
    zA5,zD2,zD1 = np.zeros((len(A5))),np.zeros(len(D2)),np.zeros((len(D1)))
    value = (np.median(abs(D1))/0.6745)*(np.sqrt(2*np.log(len(data))))
    tD5 = pywt.threshold(D5,value,'hard')
    tD4 = pywt.threshold(D4,value,'hard')
    tD3 = pywt.threshold(D3,value,'hard')
    coeffs = [zA5,tD5,tD4,tD3,zD2,zD1]
    recovered_signal = pywt.waverec(coeffs,wavelet)
    return recovered_signal

hpEEG = icaHighpass(data=rawEEG,cutoff=1,fs=cfg.fs) 

#   Computing ICA components
ica = FastICA(n_components=3, random_state=0, tol=0.0001)
comps = ica.fit_transform(hpEEG)
comps_1 = comps[:,0]
comps_2 = comps[:,1]
comps_3 = comps[:,2]



#   Computing kurtosis of ICA weights
comps_1_kurtosis = kurtosis(comps_1)
comps_2_kurtosis = kurtosis(comps_2)
comps_3_kurtosis = kurtosis(comps_3)
comps_kurtosis = np.array([comps_1_kurtosis,comps_2_kurtosis,comps_3_kurtosis])

#   Computing skewness of ICA weights
comps_1_skew = skew(comps_1)
comps_2_skew = skew(comps_2)
comps_3_skew = skew(comps_3)
comps_skew = np.array([comps_1_skew,comps_2_skew,comps_3_skew])

#   Computing sample entropy of ICA weights
comps_1_sampEN = ant.sample_entropy(comps_1)
comps_2_sampEN = ant.sample_entropy(comps_2)
comps_3_sampEN = ant.sample_entropy(comps_3)
comps_sampEN = np.array([comps_1_sampEN,comps_2_sampEN,comps_3_sampEN])

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
artf_comps = comps[:,idx_artf_comps]

""" 
Apply Discrete Wavelet Transform to epochs of ICs
remove noisy componenets within epoch
apply inverse dwt to retrieve denoised epochs
"""
wavelet = 'coif5'    
inv_comps = []
for i in range(len(idx_artf_comps)):
    inv_comps.append(dwt(artf_comps[:,i],wavelet))
inv_comps = np.array(inv_comps)


"Return dwt transformed ICs into the original windows per ICs"
for i in range(len(idx_artf_comps)):
    idx_inv = np.arange(len(inv_comps))
    comps.T[idx_artf_comps[i]] = inv_comps[idx_inv[i]]


"Recover clean signal from clean ICs"
restored = ica.inverse_transform(comps)


plt.plot(restored[:,0])
plt.show()
plt.plot(rawEEG[:,0])
plt.show()

plt.plot(restored[:,1])
plt.show()
plt.plot(rawEEG[:,1])
plt.show()

plt.plot(restored[:,2])
plt.show()
plt.plot(rawEEG[:,2])
plt.show()