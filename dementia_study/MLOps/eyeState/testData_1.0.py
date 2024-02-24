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


#  Extract features from augmented dataset
functions = featureExtraction()
lowDelta,highDelta = [0,4]
lowTheta,highTheta = [4,8]
lowAlpha,highAlpha = [8,13]
lowBeta,highBeta = [13,30]
lowGamma,highGamma = [30,44]

#    Eye open
rpDeltaFzEO = functions.relPower(fz_psd_EO.T,fs,lowDelta,highDelta,freqs,data_2d=True,dataPSD=True)
rpThetaFzEO = functions.relPower(fz_psd_EO.T,fs,lowTheta,highTheta,freqs,data_2d=True,dataPSD=True)
rpAlphaFzEO = functions.relPower(fz_psd_EO.T,fs,lowAlpha,highAlpha,freqs,data_2d=True,dataPSD=True)
rpBetaFzEO = functions.relPower(fz_psd_EO.T,fs,lowBeta,highBeta,freqs,data_2d=True,dataPSD=True)
rpGammaFzEO = functions.relPower(fz_psd_EO.T,fs,lowGamma,highGamma,freqs,data_2d=True,dataPSD=True)

rpDeltaCzEO = functions.relPower(cz_psd_EO.T,fs,lowDelta,highDelta,freqs,data_2d=True,dataPSD=True)
rpThetaCzEO = functions.relPower(cz_psd_EO.T,fs,lowTheta,highTheta,freqs,data_2d=True,dataPSD=True)
rpAlphaCzEO = functions.relPower(cz_psd_EO.T,fs,lowAlpha,highAlpha,freqs,data_2d=True,dataPSD=True)
rpBetaCzEO = functions.relPower(cz_psd_EO.T,fs,lowBeta,highBeta,freqs,data_2d=True,dataPSD=True)
rpGammaCzEO = functions.relPower(cz_psd_EO.T,fs,lowGamma,highGamma,freqs,data_2d=True,dataPSD=True)

rpDeltaPzEO = functions.relPower(pz_psd_EO.T,fs,lowDelta,highDelta,freqs,data_2d=True,dataPSD=True)
rpThetaPzEO = functions.relPower(pz_psd_EO.T,fs,lowTheta,highTheta,freqs,data_2d=True,dataPSD=True)
rpAlphaPzEO = functions.relPower(pz_psd_EO.T,fs,lowAlpha,highAlpha,freqs,data_2d=True,dataPSD=True)
rpBetaPzEO = functions.relPower(pz_psd_EO.T,fs,lowBeta,highBeta,freqs,data_2d=True,dataPSD=True)
rpGammaPzEO = functions.relPower(pz_psd_EO.T,fs,lowGamma,highGamma,freqs,data_2d=True,dataPSD=True)

rpDeltaGlobalEO = functions.relPower(global_psd_EO.T,fs,lowDelta,highDelta,freqs,data_2d=True,dataPSD=True)
rpThetaGlobalEO = functions.relPower(global_psd_EO.T,fs,lowTheta,highTheta,freqs,data_2d=True,dataPSD=True)
rpAlphaGlobalEO = functions.relPower(global_psd_EO.T,fs,lowAlpha,highAlpha,freqs,data_2d=True,dataPSD=True)
rpBetaGlobalEO = functions.relPower(global_psd_EO.T,fs,lowBeta,highBeta,freqs,data_2d=True,dataPSD=True)
rpGammaGlobalEO = functions.relPower(global_psd_EO.T,fs,lowGamma,highGamma,freqs,data_2d=True,dataPSD=True)

#    Eye close
rpDeltaFzEC = functions.relPower(fz_psd_EC.T,fs,lowDelta,highDelta,freqs,data_2d=True,dataPSD=True)
rpThetaFzEC = functions.relPower(fz_psd_EC.T,fs,lowTheta,highTheta,freqs,data_2d=True,dataPSD=True)
rpAlphaFzEC = functions.relPower(fz_psd_EC.T,fs,lowAlpha,highAlpha,freqs,data_2d=True,dataPSD=True)
rpBetaFzEC = functions.relPower(fz_psd_EC.T,fs,lowBeta,highBeta,freqs,data_2d=True,dataPSD=True)
rpGammaFzEC = functions.relPower(fz_psd_EC.T,fs,lowGamma,highGamma,freqs,data_2d=True,dataPSD=True)

rpDeltaCzEC = functions.relPower(cz_psd_EC.T,fs,lowDelta,highDelta,freqs,data_2d=True,dataPSD=True)
rpThetaCzEC = functions.relPower(cz_psd_EC.T,fs,lowTheta,highTheta,freqs,data_2d=True,dataPSD=True)
rpAlphaCzEC = functions.relPower(cz_psd_EC.T,fs,lowAlpha,highAlpha,freqs,data_2d=True,dataPSD=True)
rpBetaCzEC = functions.relPower(cz_psd_EC.T,fs,lowBeta,highBeta,freqs,data_2d=True,dataPSD=True)
rpGammaCzEC = functions.relPower(cz_psd_EC.T,fs,lowGamma,highGamma,freqs,data_2d=True,dataPSD=True)

rpDeltaPzEC = functions.relPower(pz_psd_EC.T,fs,lowDelta,highDelta,freqs,data_2d=True,dataPSD=True)
rpThetaPzEC = functions.relPower(pz_psd_EC.T,fs,lowTheta,highTheta,freqs,data_2d=True,dataPSD=True)
rpAlphaPzEC = functions.relPower(pz_psd_EC.T,fs,lowAlpha,highAlpha,freqs,data_2d=True,dataPSD=True)
rpBetaPzEC = functions.relPower(pz_psd_EC.T,fs,lowBeta,highBeta,freqs,data_2d=True,dataPSD=True)
rpGammaPzEC = functions.relPower(pz_psd_EC.T,fs,lowGamma,highGamma,freqs,data_2d=True,dataPSD=True)

rpDeltaGlobalEC = functions.relPower(global_psd_EC.T,fs,lowDelta,highDelta,freqs,data_2d=True,dataPSD=True)
rpThetaGlobalEC = functions.relPower(global_psd_EC.T,fs,lowTheta,highTheta,freqs,data_2d=True,dataPSD=True)
rpAlphaGlobalEC = functions.relPower(global_psd_EC.T,fs,lowAlpha,highAlpha,freqs,data_2d=True,dataPSD=True)
rpBetaGlobalEC = functions.relPower(global_psd_EC.T,fs,lowBeta,highBeta,freqs,data_2d=True,dataPSD=True)
rpGammaGlobalEC = functions.relPower(global_psd_EC.T,fs,lowGamma,highGamma,freqs,data_2d=True,dataPSD=True)


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
dfTest = pd.DataFrame({'fzRelativeDelta':rpDeltaFz,'fzRelativeTheta':rpThetaFz,'fzRelativeAlpha':rpAlphaFz,
                        'fzRelativeBeta':rpBetaFz,'fzRelativeGamma':rpGammaFz,
                        'czRelativeDelta':rpDeltaCz,'czRelativeTheta':rpThetaCz,'czRelativeAlpha':rpAlphaCz,
                        'czRelativeBeta':rpBetaCz,'czRelativeGamma':rpGammaCz,
                        'pzRelativeDelta':rpDeltaPz,'pzRelativeTheta':rpThetaPz,'pzRelativeAlpha':rpAlphaPz,
                        'pzRelativeBeta':rpBetaPz,'pzRelativeGamma':rpGammaPz,
                        'globalRelativeDelta':rpDeltaGlobal,'globalRelativeTheta':rpThetaGlobal,'globalRelativeAlpha':rpAlphaGlobal,
                        'globalRelativeBeta':rpBetaGlobal,'globalRelativeGamma':rpGammaGlobal,
                        'labels':labels[:,0]})
dfTest = dfTest.sample(frac=1).reset_index()
dfTest.to_csv(dataPath + 'originalTestData.csv',index=False)