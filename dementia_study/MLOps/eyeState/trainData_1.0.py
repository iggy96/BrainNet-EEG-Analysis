"""
Generation of eye open/close dataset
Eyes Open and Eyes Close Activity Recognition Using EEG Signals:
For activity recognition, a freely online available EEG-based motor movement and imaginary dataset provided by PhysioNet BCI [12] has been used. 
EEG signals were recorded from all 64 channels for 1 min. 
Two baseline tasks, eyes open (EO) and eyes close (EC) resting state have been used to collect the data from 109 users and are considered in this work. 
In order to detect each activity accurately, the EEG data has been segmented into 10 s. 
Thus, a total of 1308 EEG files (i.e. 654 for EC and 654 for EO) have been created for analysis
link to paper: https://link-springer-com.proxy.lib.sfu.ca/content/pdf/10.1007/978-981-10-9059-2.pdf
link to dataset: https://physionet.org/content/eegmmidb/1.0.0/
"""


import sys
sys.path.insert(1, '/Users/joshuaighalo/Documents/GitHub/eegDementia')
from fn_cfg import *
import params as cfg

dataPath = '/Users/joshuaighalo/Documents/BrainNet/Projects/Workspace/Datasets/eyeState/trainDataset/'

def psdExtractor(file,path):
    subfolder = file[:-7]
    directory = path + '/' + subfolder + '/' + file
    edf_file = mne.io.read_raw_edf(directory)
    raw_data = edf_file.get_data()
    info = edf_file.info
    channelNames = info['ch_names']
    fs = int(info['sfreq'])
    chansEEG = raw_data.T
    ts = (np.arange(0,len(chansEEG)/fs,1/fs)).reshape(len(np.arange(0,len(chansEEG)/fs,1/fs)),1)
    icaEEG = ICA(chansEEG,fs)
    icaEEG = icaEEG[:,[33,10,50]]
    globalEEG = np.mean(icaEEG,axis=1)
    icaEEG = np.hstack((icaEEG,globalEEG.reshape(len(globalEEG),1)))
    freqs,PSD = psd(icaEEG,fs,data_2D=True)
    freqs,PSD = freqs[0],PSD.T
    def addNan(x):
        noEpochs = 4881
        lenRow = len(x)
        lenCol = len(x.T)
        remainRows = int(noEpochs-lenRow)
        nansValues = np.full((remainRows,lenCol),np.nan)
        return np.vstack((x,nansValues))
    PSD = addNan(PSD)
    return PSD,freqs

# extract all 01.edf files from their folders
def extractFiles(path,searchstring):
    files = []
    for r, d, f in os.walk(path):
        for file in f:
            if searchstring in file:
                files.append(file)
    # remove files ending with '.event'
    files = [x for x in files if not x.endswith('.event')]
    return files
path = directory = '/Users/joshuaighalo/Downloads/brainNet_datasets/eye open & close dataset'
files_EO = extractFiles(path,'01.edf')
files_EC = extractFiles(path,'02.edf')

#   extract features for both conditions
psd_EO,psd_EC,freqs = [],[],[]
for i in range(len(files_EO)):
    PSD,freqs = psdExtractor(files_EO[i],directory)
    psd_EO.append(PSD)
psd_EO,freqs = np.array(psd_EO),np.array(freqs)
for i in range(len(files_EC)):
    PSD,freqs = psdExtractor(files_EC[i],directory)
    psd_EC.append(PSD)
psd_EC,freqs = np.array(psd_EC),np.array(freqs)
fz_psd_EO = psd_EO[:,:,0]
fz_psd_EC = psd_EC[:,:,0]
cz_psd_EO = psd_EO[:,:,1]
cz_psd_EC = psd_EC[:,:,1]
pz_psd_EO = psd_EO[:,:,2]
pz_psd_EC = psd_EC[:,:,2]
global_psd_EO = psd_EO[:,:,3]
global_psd_EC = psd_EC[:,:,3]

#  Plot the grand PSDs
grand_psd_EO = np.mean(psd_EO,axis=0)
grand_psd_EC = np.mean(psd_EC,axis=0)
fig,ax = plt.subplots(1,4,figsize=(30,15))
ax[0].plot(freqs,grand_psd_EO[:,0],color='blue',label='EO')
ax[0].plot(freqs,grand_psd_EC[:,0],color='red',label='EC')
ax[0].set_xlim([0,50])
ax[0].set_title('Fz')
ax[1].plot(freqs,grand_psd_EO[:,1],color='blue',label='EO')
ax[1].plot(freqs,grand_psd_EC[:,1],color='red',label='EC')
ax[1].set_xlim([0,50])
ax[1].set_title('Cz')
ax[2].plot(freqs,grand_psd_EO[:,2],color='blue',label='EO')
ax[2].plot(freqs,grand_psd_EC[:,2],color='red',label='EC')
ax[2].set_xlim([0,50])
ax[2].set_title('Pz')
ax[3].plot(freqs,grand_psd_EO[:,3],color='blue',label='EO')
ax[3].plot(freqs,grand_psd_EC[:,3],color='red',label='EC')
ax[3].set_xlim([0,50])
ax[3].set_title('Global')
ax[0].legend()
ax[0].set_xlabel('Frequency (Hz)')
ax[0].set_ylabel('Power (uV^2/Hz)')
plt.tight_layout()
plt.show()

#% Data augmentation
numSamples = 2000

#    Quantize the data
quanFzEO = tsaug.Quantize(repeats=int(numSamples/len(fz_psd_EO))).augment(fz_psd_EO)
quanFzEC = tsaug.Quantize(repeats=int(numSamples/len(fz_psd_EC))).augment(fz_psd_EC)
quanCzEO = tsaug.Quantize(repeats=int(numSamples/len(cz_psd_EO))).augment(cz_psd_EO)
quanCzEC = tsaug.Quantize(repeats=int(numSamples/len(cz_psd_EC))).augment(cz_psd_EC)
quanPzEO = tsaug.Quantize(repeats=int(numSamples/len(pz_psd_EO))).augment(pz_psd_EO)
quanPzEC = tsaug.Quantize(repeats=int(numSamples/len(pz_psd_EC))).augment(pz_psd_EC)
quanGlobalEO = tsaug.Quantize(repeats=int(numSamples/len(global_psd_EO))).augment(global_psd_EO)
quanGlobalEC = tsaug.Quantize(repeats=int(numSamples/len(global_psd_EC))).augment(global_psd_EC)

# Validate augmented dataset
fig,ax = plt.subplots(1,4,figsize=(30,15))
ax[0].plot(freqs,np.nanmean(quanFzEO,axis=0),color='blue',linestyle='dashdot',label='Quantize EO')
ax[0].plot(freqs,np.nanmean(quanFzEC,axis=0),color='red',linestyle='dashdot',label='Quantize EC')
ax[0].plot(freqs,np.nanmean(fz_psd_EO,axis=0),color='green',label='EO')
ax[0].plot(freqs,np.nanmean(fz_psd_EC,axis=0),color='orange',label='EC')
ax[0].set_xlim([0,44])
ax[0].set_title('Fz')
ax[1].plot(freqs,np.nanmean(quanCzEO,axis=0),color='blue',linestyle='dashdot',label='Quantize EO')
ax[1].plot(freqs,np.nanmean(quanCzEC,axis=0),color='red',linestyle='dashdot',label='Quantize EC')
ax[1].plot(freqs,np.nanmean(cz_psd_EO,axis=0),color='green',label='EO')
ax[1].plot(freqs,np.nanmean(cz_psd_EC,axis=0),color='orange',label='EC')
ax[1].set_xlim([0,44])
ax[1].set_title('Cz')
ax[2].plot(freqs,np.nanmean(quanPzEO,axis=0),color='blue',linestyle='dashdot',label='Quantize EO')
ax[2].plot(freqs,np.nanmean(quanPzEC,axis=0),color='red',linestyle='dashdot',label='Quantize EC')
ax[2].plot(freqs,np.nanmean(pz_psd_EO,axis=0),color='green',label='EO')
ax[2].plot(freqs,np.nanmean(pz_psd_EC,axis=0),color='orange',label='EC')
ax[2].set_xlim([0,44])
ax[2].set_title('Pz')
ax[3].plot(freqs,np.nanmean(quanGlobalEO,axis=0),color='blue',linestyle='dashdot',label='Quantize EO')
ax[3].plot(freqs,np.nanmean(quanGlobalEC,axis=0),color='red',linestyle='dashdot',label='Quantize EC')
ax[3].plot(freqs,np.nanmean(global_psd_EO,axis=0),color='green',label='EO')
ax[3].plot(freqs,np.nanmean(global_psd_EC,axis=0),color='orange',label='EC')
ax[3].set_xlim([0,44])
ax[3].set_title('Global')
ax[0].legend()
ax[1].legend()
ax[2].legend()
ax[3].legend()
plt.show()

#  Extract features from augmented dataset
functions = featureExtraction()
lowDelta,highDelta = [0,4]
lowTheta,highTheta = [4,8]
lowAlpha,highAlpha = [8,13]
lowBeta,highBeta = [13,30]
lowGamma,highGamma = [30,44]

#    Eye open
rpDeltaFzEO = functions.relPower(quanFzEO.T,fs,lowDelta,highDelta,freqs,data_2d=True,dataPSD=True)
rpThetaFzEO = functions.relPower(quanFzEO.T,fs,lowTheta,highTheta,freqs,data_2d=True,dataPSD=True)
rpAlphaFzEO = functions.relPower(quanFzEO.T,fs,lowAlpha,highAlpha,freqs,data_2d=True,dataPSD=True)
rpBetaFzEO = functions.relPower(quanFzEO.T,fs,lowBeta,highBeta,freqs,data_2d=True,dataPSD=True)
rpGammaFzEO = functions.relPower(quanFzEO.T,fs,lowGamma,highGamma,freqs,data_2d=True,dataPSD=True)

rpDeltaCzEO = functions.relPower(quanCzEO.T,fs,lowDelta,highDelta,freqs,data_2d=True,dataPSD=True)
rpThetaCzEO = functions.relPower(quanCzEO.T,fs,lowTheta,highTheta,freqs,data_2d=True,dataPSD=True)
rpAlphaCzEO = functions.relPower(quanCzEO.T,fs,lowAlpha,highAlpha,freqs,data_2d=True,dataPSD=True)
rpBetaCzEO = functions.relPower(quanCzEO.T,fs,lowBeta,highBeta,freqs,data_2d=True,dataPSD=True)
rpGammaCzEO = functions.relPower(quanCzEO.T,fs,lowGamma,highGamma,freqs,data_2d=True,dataPSD=True)

rpDeltaPzEO = functions.relPower(quanPzEO.T,fs,lowDelta,highDelta,freqs,data_2d=True,dataPSD=True)
rpThetaPzEO = functions.relPower(quanPzEO.T,fs,lowTheta,highTheta,freqs,data_2d=True,dataPSD=True)
rpAlphaPzEO = functions.relPower(quanPzEO.T,fs,lowAlpha,highAlpha,freqs,data_2d=True,dataPSD=True)
rpBetaPzEO = functions.relPower(quanPzEO.T,fs,lowBeta,highBeta,freqs,data_2d=True,dataPSD=True)
rpGammaPzEO = functions.relPower(quanPzEO.T,fs,lowGamma,highGamma,freqs,data_2d=True,dataPSD=True)

rpDeltaGlobalEO = functions.relPower(quanGlobalEO.T,fs,lowDelta,highDelta,freqs,data_2d=True,dataPSD=True)
rpThetaGlobalEO = functions.relPower(quanGlobalEO.T,fs,lowTheta,highTheta,freqs,data_2d=True,dataPSD=True)
rpAlphaGlobalEO = functions.relPower(quanGlobalEO.T,fs,lowAlpha,highAlpha,freqs,data_2d=True,dataPSD=True)
rpBetaGlobalEO = functions.relPower(quanGlobalEO.T,fs,lowBeta,highBeta,freqs,data_2d=True,dataPSD=True)
rpGammaGlobalEO = functions.relPower(quanGlobalEO.T,fs,lowGamma,highGamma,freqs,data_2d=True,dataPSD=True)

#    Eye closed
rpDeltaFzEC = functions.relPower(quanFzEC.T,fs,lowDelta,highDelta,freqs,data_2d=True,dataPSD=True)
rpThetaFzEC = functions.relPower(quanFzEC.T,fs,lowTheta,highTheta,freqs,data_2d=True,dataPSD=True)
rpAlphaFzEC = functions.relPower(quanFzEC.T,fs,lowAlpha,highAlpha,freqs,data_2d=True,dataPSD=True)
rpBetaFzEC = functions.relPower(quanFzEC.T,fs,lowBeta,highBeta,freqs,data_2d=True,dataPSD=True)
rpGammaFzEC = functions.relPower(quanFzEC.T,fs,lowGamma,highGamma,freqs,data_2d=True,dataPSD=True)

rpDeltaCzEC = functions.relPower(quanCzEC.T,fs,lowDelta,highDelta,freqs,data_2d=True,dataPSD=True)
rpThetaCzEC = functions.relPower(quanCzEC.T,fs,lowTheta,highTheta,freqs,data_2d=True,dataPSD=True)
rpAlphaCzEC = functions.relPower(quanCzEC.T,fs,lowAlpha,highAlpha,freqs,data_2d=True,dataPSD=True)
rpBetaCzEC = functions.relPower(quanCzEC.T,fs,lowBeta,highBeta,freqs,data_2d=True,dataPSD=True)
rpGammaCzEC = functions.relPower(quanCzEC.T,fs,lowGamma,highGamma,freqs,data_2d=True,dataPSD=True)

rpDeltaPzEC = functions.relPower(quanPzEC.T,fs,lowDelta,highDelta,freqs,data_2d=True,dataPSD=True)
rpThetaPzEC = functions.relPower(quanPzEC.T,fs,lowTheta,highTheta,freqs,data_2d=True,dataPSD=True)
rpAlphaPzEC = functions.relPower(quanPzEC.T,fs,lowAlpha,highAlpha,freqs,data_2d=True,dataPSD=True)
rpBetaPzEC = functions.relPower(quanPzEC.T,fs,lowBeta,highBeta,freqs,data_2d=True,dataPSD=True)
rpGammaPzEC = functions.relPower(quanPzEC.T,fs,lowGamma,highGamma,freqs,data_2d=True,dataPSD=True)

rpDeltaGlobalEC = functions.relPower(quanGlobalEC.T,fs,lowDelta,highDelta,freqs,data_2d=True,dataPSD=True)
rpThetaGlobalEC = functions.relPower(quanGlobalEC.T,fs,lowTheta,highTheta,freqs,data_2d=True,dataPSD=True)
rpAlphaGlobalEC = functions.relPower(quanGlobalEC.T,fs,lowAlpha,highAlpha,freqs,data_2d=True,dataPSD=True)
rpBetaGlobalEC = functions.relPower(quanGlobalEC.T,fs,lowBeta,highBeta,freqs,data_2d=True,dataPSD=True)
rpGammaGlobalEC = functions.relPower(quanGlobalEC.T,fs,lowGamma,highGamma,freqs,data_2d=True,dataPSD=True)


#   generate labels
labels_EO = np.zeros((len(rpDeltaFzEO),1))
labels_EC = np.ones((len(rpDeltaFzEC),1))

#   concatenate data
rpDeltaFz, rpThetaFz, rpAlphaFz, rpBetaFz, rpGammaFz = np.concatenate((rpDeltaFzEO,rpDeltaFzEC)), np.concatenate((rpThetaFzEO,rpThetaFzEC)), np.concatenate((rpAlphaFzEO,rpAlphaFzEC)), np.concatenate((rpBetaFzEO,rpBetaFzEC)), np.concatenate((rpGammaFzEO,rpGammaFzEC))
rpDeltaCz, rpThetaCz, rpAlphaCz, rpBetaCz, rpGammaCz = np.concatenate((rpDeltaCzEO,rpDeltaCzEC)), np.concatenate((rpThetaCzEO,rpThetaCzEC)), np.concatenate((rpAlphaCzEO,rpAlphaCzEC)), np.concatenate((rpBetaCzEO,rpBetaCzEC)), np.concatenate((rpGammaCzEO,rpGammaCzEC))
rpDeltaPz, rpThetaPz, rpAlphaPz, rpBetaPz, rpGammaPz = np.concatenate((rpDeltaPzEO,rpDeltaPzEC)), np.concatenate((rpThetaPzEO,rpThetaPzEC)), np.concatenate((rpAlphaPzEO,rpAlphaPzEC)), np.concatenate((rpBetaPzEO,rpBetaPzEC)), np.concatenate((rpGammaPzEO,rpGammaPzEC))
rpDeltaGlobal, rpThetaGlobal, rpAlphaGlobal, rpBetaGlobal, rpGammaGlobal = np.concatenate((rpDeltaGlobalEO,rpDeltaGlobalEC)), np.concatenate((rpThetaGlobalEO,rpThetaGlobalEC)), np.concatenate((rpAlphaGlobalEO,rpAlphaGlobalEC)), np.concatenate((rpBetaGlobalEO,rpBetaGlobalEC)), np.concatenate((rpGammaGlobalEO,rpGammaGlobalEC))
# concatenate labels
labels = np.concatenate((labels_EO,labels_EC))

# reshape Dataframes and concatenate
dfTrain = pd.DataFrame({'fzRelativeDelta':rpDeltaFz,'fzRelativeTheta':rpThetaFz,'fzRelativeAlpha':rpAlphaFz,
                        'fzRelativeBeta':rpBetaFz,'fzRelativeGamma':rpGammaFz,
                        'czRelativeDelta':rpDeltaCz,'czRelativeTheta':rpThetaCz,'czRelativeAlpha':rpAlphaCz,
                        'czRelativeBeta':rpBetaCz,'czRelativeGamma':rpGammaCz,
                        'pzRelativeDelta':rpDeltaPz,'pzRelativeTheta':rpThetaPz,'pzRelativeAlpha':rpAlphaPz,
                        'pzRelativeBeta':rpBetaPz,'pzRelativeGamma':rpGammaPz,
                        'globalRelativeDelta':rpDeltaGlobal,'globalRelativeTheta':rpThetaGlobal,'globalRelativeAlpha':rpAlphaGlobal,
                        'globalRelativeBeta':rpBetaGlobal,'globalRelativeGamma':rpGammaGlobal,
                        'labels':labels[:,0]})
dfTrain = dfTrain.sample(frac=1).reset_index()
dfTrain.to_csv(dataPath + 'trainData.csv',index=False)
