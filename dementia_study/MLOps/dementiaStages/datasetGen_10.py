from ds_helper import*
import ds_params as cfg
import sys
#sys.path.insert(1, '/Users/joshuaighalo/Documents/GitHub/eegDementia')
#from eeg_helper import*
       
npy_dir = cfg.scans_dir       
version,sf,line,high,low = cfg.device_version,cfg.fs,cfg.line_freq,cfg.highpass_freq,cfg.lowpass_freq
dataPath = cfg.path_to_dataset

#%% filenames of classes
baseND,sixND,twelveND = np.load(npy_dir+'baselineNoDementia.npy'),np.load(npy_dir+'6NoDementia.npy'),np.load(npy_dir+'12NoDementia.npy')
baseMMD,sixMMD,twelveMMD = np.load(npy_dir+'baselineMildModDementia.npy'),np.load(npy_dir+'6MildModDementia.npy'),np.load(npy_dir+'12MildModDementia.npy')
baseSD,sixSD,twelveSD = np.load(npy_dir+'baselineSevereDementia.npy'),np.load(npy_dir+'6SevereDementia.npy'),np.load(npy_dir+'12SevereDementia.npy')

#%% extract raw eegs
eeg_baseND,eeg_sixND,eeg_twelveND = extractEEG(baseND,version,dataPath,sf,line,high,low),extractEEG(sixND,version,dataPath,sf,line,high,low),extractEEG(twelveND,version,dataPath,sf,line,high,low)
eeg_baseMMD,eeg_sixMMD,eeg_twelveMMD = extractEEG(baseMMD,version,dataPath,sf,line,high,low),extractEEG(sixMMD,version,dataPath,sf,line,high,low),extractEEG(twelveMMD,version,dataPath,sf,line,high,low)
eeg_baseSD,eeg_sixSD,eeg_twelveSD = extractEEG(baseSD,version,dataPath,sf,line,high,low),extractEEG(sixSD,version,dataPath,sf,line,high,low),extractEEG(twelveSD,version,dataPath,sf,line,high,low)
dt = 1/sf
stop = len(eeg_baseND[0,:,0])/sf
time_s = (np.arange(0,stop,dt)).reshape(len(np.arange(0,stop,dt)),1)

# %% remove artifacts
# output format: (numsubjects,numchannels,numwindows,windowlength)
threshold_value,window_size,step_size,channels,choice_numwindows = 10,0.5,0.5,['Fz','Cz','Pz'],618
ar_baseND = multipleArtfRemoval(eeg_baseND,threshold_value,time_s,window_size,step_size,choice_numwindows,channels,"baseline No Dementia")
ar_6ND = multipleArtfRemoval(eeg_sixND,threshold_value,time_s,window_size,step_size,choice_numwindows,channels,"6 months No Dementia")
ar_12ND = multipleArtfRemoval(eeg_twelveND,threshold_value,time_s,window_size,step_size,choice_numwindows,channels,"12 months No Dementia")
ar_baseMMD = multipleArtfRemoval(eeg_baseMMD,threshold_value,time_s,window_size,step_size,choice_numwindows,channels,"baseline Mild Mod Dementia")
ar_6MMD = multipleArtfRemoval(eeg_sixMMD,threshold_value,time_s,window_size,step_size,choice_numwindows,channels,"6 months Mild Mod Dementia")
ar_12MMD = multipleArtfRemoval(eeg_twelveMMD,threshold_value,time_s,window_size,step_size,choice_numwindows,channels,"12 months Mild Mod Dementia")
ar_baseSD = multipleArtfRemoval(eeg_baseSD,threshold_value,time_s,window_size,step_size,choice_numwindows,channels,"baseline Severe Dementia")
ar_6SD = multipleArtfRemoval(eeg_sixSD,threshold_value,time_s,window_size,step_size,choice_numwindows,channels,"6 months Severe Dementia")
ar_12SD = multipleArtfRemoval(eeg_twelveSD,threshold_value,time_s,window_size,step_size,choice_numwindows,channels,"12 months Severe Dementia")

#%% change output shape to (numsubjects,numchannels,windowlength,numwindows)
arr_baseND,arr_6ND,arr_12ND = np.transpose(ar_baseND,(0,1,3,2)),np.transpose(ar_6ND,(0,1,3,2)),np.transpose(ar_12ND,(0,1,3,2))
arr_baseMMD,arr_6MMD,arr_12MMD = np.transpose(ar_baseMMD,(0,1,3,2)),np.transpose(ar_6MMD,(0,1,3,2)),np.transpose(ar_12MMD,(0,1,3,2))
arr_baseSD,arr_6SD,arr_12SD = np.transpose(ar_baseSD,(0,1,3,2)),np.transpose(ar_6SD,(0,1,3,2)),np.transpose(ar_12SD,(0,1,3,2))

#%% compute PSD
freqs,psd_baseND = multiplePSD(arr_baseND,sf)
freqs,psd_6ND = multiplePSD(arr_6ND,sf)
freqs,psd_12ND = multiplePSD(arr_12ND,sf)
freqs,psd_baseMMD = multiplePSD(arr_baseMMD,sf)
freqs,psd_6MMD = multiplePSD(arr_6MMD,sf)
freqs,psd_12MMD = multiplePSD(arr_12MMD,sf)
freqs,psd_baseSD = multiplePSD(arr_baseSD,sf)
freqs,psd_6SD = multiplePSD(arr_6SD,sf)
freqs,psd_12SD = multiplePSD(arr_12SD,sf)
freqs = freqs[0,0,0,:]
#%%
psd_baseND,psd_6ND,psd_12ND = np.nanmean(psd_baseND,axis=2),np.nanmean(psd_6ND,axis=2),np.nanmean(psd_12ND,axis=2)
psd_baseMMD,psd_6MMD,psd_12MMD = np.nanmean(psd_baseMMD,axis=2),np.nanmean(psd_6MMD,axis=2),np.nanmean(psd_12MMD,axis=2)
psd_baseSD,psd_6SD,psd_12SD = np.nanmean(psd_baseSD,axis=2),np.nanmean(psd_6SD,axis=2),np.nanmean(psd_12SD,axis=2)

#%% plot train set
fzPSDBND,czPSDBND,pzPSDBND = (np.mean(psd_baseND,axis=0))[0],(np.mean(psd_baseND,axis=0))[1],(np.mean(psd_baseND,axis=0))[2]
fzPSDBMMD,czPSDBMMD,pzPSDBMMD = (np.mean(psd_baseMMD,axis=0))[0],(np.mean(psd_baseMMD,axis=0))[1],(np.mean(psd_baseMMD,axis=0))[2]
fzPSDBSD,czPSDBSD,pzPSDBSD = (np.mean(psd_baseSD,axis=0))[0],(np.mean(psd_baseSD,axis=0))[1],(np.mean(psd_baseSD,axis=0))[2]
fig,ax = plt.subplots(1,3,figsize=(20,10))
ax[0].plot(freqs,fzPSDBND,label="baseline ND")
ax[0].plot(freqs,fzPSDBMMD,label="baseline MMD")
ax[0].plot(freqs,fzPSDBSD,label="baseline SD")
ax[1].plot(freqs,czPSDBND,label="baseline ND")
ax[1].plot(freqs,czPSDBMMD,label="baseline MMD")
ax[1].plot(freqs,czPSDBSD,label="baseline SD")
ax[2].plot(freqs,pzPSDBND,label="baseline ND")
ax[2].plot(freqs,pzPSDBMMD,label="baseline MMD")
ax[2].plot(freqs,pzPSDBSD,label="baseline SD")
ax[0].set_title("Fz")
ax[1].set_title("Cz")
ax[2].set_title("Pz")
ax[0].set_ylabel("PSD(dB)")
ax[1].set_ylabel("PSD(dB)")
ax[2].set_ylabel("PSD(dB)")
ax[0].set_xlabel("Frequency(Hz)")
ax[1].set_xlabel("Frequency(Hz)")
ax[2].set_xlabel("Frequency(Hz)")
ax[0].legend()
ax[1].legend()
ax[2].legend()
ax[0].set_xlim(0,50)
ax[1].set_xlim(0,50)
ax[2].set_xlim(0,50)
plt.show()
 

#%% Full band power
absFullBaseND = multipleAbsoluteBandPower((np.transpose(psd_baseND,(0,2,1))),freqs,cfg.delta[0],cfg.gamma[1])
absFull6ND = multipleAbsoluteBandPower((np.transpose(psd_6ND,(0,2,1))),freqs,cfg.delta[0],cfg.gamma[1])
absFull12ND = multipleAbsoluteBandPower((np.transpose(psd_12ND,(0,2,1))),freqs,cfg.delta[0],cfg.gamma[1])
absFullBaseMMD = multipleAbsoluteBandPower((np.transpose(psd_baseMMD,(0,2,1))),freqs,cfg.delta[0],cfg.gamma[1])
absFull6MMD = multipleAbsoluteBandPower((np.transpose(psd_6MMD,(0,2,1))),freqs,cfg.delta[0],cfg.gamma[1])
absFull12MMD = multipleAbsoluteBandPower((np.transpose(psd_12MMD,(0,2,1))),freqs,cfg.delta[0],cfg.gamma[1])
absFullBaseSD = multipleAbsoluteBandPower((np.transpose(psd_baseSD,(0,2,1))),freqs,cfg.delta[0],cfg.gamma[1])
absFull6SD = multipleAbsoluteBandPower((np.transpose(psd_6SD,(0,2,1))),freqs,cfg.delta[0],cfg.gamma[1])
absFull12SD = multipleAbsoluteBandPower((np.transpose(psd_12SD,(0,2,1))),freqs,cfg.delta[0],cfg.gamma[1])

#%% Relative band power: Full delta, theta, alpha, beta, gamma
relDeltaBaseND = multipleRelativeBandPower((np.transpose(psd_baseND,(0,2,1))),freqs,cfg.delta[0],cfg.delta[1])
relThetaBaseND = multipleRelativeBandPower((np.transpose(psd_baseND,(0,2,1))),freqs,cfg.theta[0],cfg.theta[1])
relAlphaBaseND = multipleRelativeBandPower((np.transpose(psd_baseND,(0,2,1))),freqs,cfg.alpha[0],cfg.alpha[1])
relBetaBaseND = multipleRelativeBandPower((np.transpose(psd_baseND,(0,2,1))),freqs,cfg.beta[0],cfg.beta[1])
relGammaBaseND = multipleRelativeBandPower((np.transpose(psd_baseND,(0,2,1))),freqs,cfg.gamma[0],cfg.gamma[1])
relDelta6ND = multipleRelativeBandPower((np.transpose(psd_6ND,(0,2,1))),freqs,cfg.delta[0],cfg.delta[1])
relTheta6ND = multipleRelativeBandPower((np.transpose(psd_6ND,(0,2,1))),freqs,cfg.theta[0],cfg.theta[1])
relAlpha6ND = multipleRelativeBandPower((np.transpose(psd_6ND,(0,2,1))),freqs,cfg.alpha[0],cfg.alpha[1])
relBeta6ND = multipleRelativeBandPower((np.transpose(psd_6ND,(0,2,1))),freqs,cfg.beta[0],cfg.beta[1])
relGamma6ND = multipleRelativeBandPower((np.transpose(psd_6ND,(0,2,1))),freqs,cfg.gamma[0],cfg.gamma[1])
relDelta12ND = multipleRelativeBandPower((np.transpose(psd_12ND,(0,2,1))),freqs,cfg.delta[0],cfg.delta[1])
relTheta12ND = multipleRelativeBandPower((np.transpose(psd_12ND,(0,2,1))),freqs,cfg.theta[0],cfg.theta[1])
relAlpha12ND = multipleRelativeBandPower((np.transpose(psd_12ND,(0,2,1))),freqs,cfg.alpha[0],cfg.alpha[1])
relBeta12ND = multipleRelativeBandPower((np.transpose(psd_12ND,(0,2,1))),freqs,cfg.beta[0],cfg.beta[1])
relGamma12ND = multipleRelativeBandPower((np.transpose(psd_12ND,(0,2,1))),freqs,cfg.gamma[0],cfg.gamma[1])

relDeltaBaseMMD = multipleRelativeBandPower((np.transpose(psd_baseMMD,(0,2,1))),freqs,cfg.delta[0],cfg.delta[1])
relThetaBaseMMD = multipleRelativeBandPower((np.transpose(psd_baseMMD,(0,2,1))),freqs,cfg.theta[0],cfg.theta[1])
relAlphaBaseMMD = multipleRelativeBandPower((np.transpose(psd_baseMMD,(0,2,1))),freqs,cfg.alpha[0],cfg.alpha[1])
relBetaBaseMMD = multipleRelativeBandPower((np.transpose(psd_baseMMD,(0,2,1))),freqs,cfg.beta[0],cfg.beta[1])
relGammaBaseMMD = multipleRelativeBandPower((np.transpose(psd_baseMMD,(0,2,1))),freqs,cfg.gamma[0],cfg.gamma[1])
relDelta6MMD = multipleRelativeBandPower((np.transpose(psd_6MMD,(0,2,1))),freqs,cfg.delta[0],cfg.delta[1])
relTheta6MMD = multipleRelativeBandPower((np.transpose(psd_6MMD,(0,2,1))),freqs,cfg.theta[0],cfg.theta[1])
relAlpha6MMD = multipleRelativeBandPower((np.transpose(psd_6MMD,(0,2,1))),freqs,cfg.alpha[0],cfg.alpha[1])
relBeta6MMD = multipleRelativeBandPower((np.transpose(psd_6MMD,(0,2,1))),freqs,cfg.beta[0],cfg.beta[1])
relGamma6MMD = multipleRelativeBandPower((np.transpose(psd_6MMD,(0,2,1))),freqs,cfg.gamma[0],cfg.gamma[1])
relDelta12MMD = multipleRelativeBandPower((np.transpose(psd_12MMD,(0,2,1))),freqs,cfg.delta[0],cfg.delta[1])
relTheta12MMD = multipleRelativeBandPower((np.transpose(psd_12MMD,(0,2,1))),freqs,cfg.theta[0],cfg.theta[1])
relAlpha12MMD = multipleRelativeBandPower((np.transpose(psd_12MMD,(0,2,1))),freqs,cfg.alpha[0],cfg.alpha[1])
relBeta12MMD = multipleRelativeBandPower((np.transpose(psd_12MMD,(0,2,1))),freqs,cfg.beta[0],cfg.beta[1])
relGamma12MMD = multipleRelativeBandPower((np.transpose(psd_12MMD,(0,2,1))),freqs,cfg.gamma[0],cfg.gamma[1])

relDeltaBaseSD = multipleRelativeBandPower((np.transpose(psd_baseSD,(0,2,1))),freqs,cfg.delta[0],cfg.delta[1])
relThetaBaseSD = multipleRelativeBandPower((np.transpose(psd_baseSD,(0,2,1))),freqs,cfg.theta[0],cfg.theta[1])
relAlphaBaseSD = multipleRelativeBandPower((np.transpose(psd_baseSD,(0,2,1))),freqs,cfg.alpha[0],cfg.alpha[1])
relBetaBaseSD = multipleRelativeBandPower((np.transpose(psd_baseSD,(0,2,1))),freqs,cfg.beta[0],cfg.beta[1])
relGammaBaseSD = multipleRelativeBandPower((np.transpose(psd_baseSD,(0,2,1))),freqs,cfg.gamma[0],cfg.gamma[1])
relDelta6SD = multipleRelativeBandPower((np.transpose(psd_6SD,(0,2,1))),freqs,cfg.delta[0],cfg.delta[1])
relTheta6SD = multipleRelativeBandPower((np.transpose(psd_6SD,(0,2,1))),freqs,cfg.theta[0],cfg.theta[1])
relAlpha6SD = multipleRelativeBandPower((np.transpose(psd_6SD,(0,2,1))),freqs,cfg.alpha[0],cfg.alpha[1])
relBeta6SD = multipleRelativeBandPower((np.transpose(psd_6SD,(0,2,1))),freqs,cfg.beta[0],cfg.beta[1])
relGamma6SD = multipleRelativeBandPower((np.transpose(psd_6SD,(0,2,1))),freqs,cfg.gamma[0],cfg.gamma[1])
relDelta12SD = multipleRelativeBandPower((np.transpose(psd_12SD,(0,2,1))),freqs,cfg.delta[0],cfg.delta[1])
relTheta12SD = multipleRelativeBandPower((np.transpose(psd_12SD,(0,2,1))),freqs,cfg.theta[0],cfg.theta[1])
relAlpha12SD = multipleRelativeBandPower((np.transpose(psd_12SD,(0,2,1))),freqs,cfg.alpha[0],cfg.alpha[1])
relBeta12SD = multipleRelativeBandPower((np.transpose(psd_12SD,(0,2,1))),freqs,cfg.beta[0],cfg.beta[1])
relGamma12SD = multipleRelativeBandPower((np.transpose(psd_12SD,(0,2,1))),freqs,cfg.gamma[0],cfg.gamma[1])
 
# %% Peak Power: Full delta, theta, alpha, beta, gamma
peakDeltaBaseND = multiplePeakPower((np.transpose(psd_baseND,(0,2,1))),freqs,cfg.delta[0],cfg.delta[1])
peakThetaBaseND = multiplePeakPower((np.transpose(psd_baseND,(0,2,1))),freqs,cfg.theta[0],cfg.theta[1]) 
peakAlphaBaseND = multiplePeakPower((np.transpose(psd_baseND,(0,2,1))),freqs,cfg.alpha[0],cfg.alpha[1])
peakBetaBaseND = multiplePeakPower((np.transpose(psd_baseND,(0,2,1))),freqs,cfg.beta[0],cfg.beta[1])
peakGammaBaseND = multiplePeakPower((np.transpose(psd_baseND,(0,2,1))),freqs,cfg.gamma[0],cfg.gamma[1])
peakDelta6ND = multiplePeakPower((np.transpose(psd_6ND,(0,2,1))),freqs,cfg.delta[0],cfg.delta[1])
peakTheta6ND = multiplePeakPower((np.transpose(psd_6ND,(0,2,1))),freqs,cfg.theta[0],cfg.theta[1])
peakAlpha6ND = multiplePeakPower((np.transpose(psd_6ND,(0,2,1))),freqs,cfg.alpha[0],cfg.alpha[1])
peakBeta6ND = multiplePeakPower((np.transpose(psd_6ND,(0,2,1))),freqs,cfg.beta[0],cfg.beta[1])
peakGamma6ND = multiplePeakPower((np.transpose(psd_6ND,(0,2,1))),freqs,cfg.gamma[0],cfg.gamma[1])
peakDelta12ND = multiplePeakPower((np.transpose(psd_12ND,(0,2,1))),freqs,cfg.delta[0],cfg.delta[1])
peakTheta12ND = multiplePeakPower((np.transpose(psd_12ND,(0,2,1))),freqs,cfg.theta[0],cfg.theta[1])
peakAlpha12ND = multiplePeakPower((np.transpose(psd_12ND,(0,2,1))),freqs,cfg.alpha[0],cfg.alpha[1])
peakBeta12ND = multiplePeakPower((np.transpose(psd_12ND,(0,2,1))),freqs,cfg.beta[0],cfg.beta[1])
peakGamma12ND = multiplePeakPower((np.transpose(psd_12ND,(0,2,1))),freqs,cfg.gamma[0],cfg.gamma[1])

peakDeltaBaseMMD = multiplePeakPower((np.transpose(psd_baseMMD,(0,2,1))),freqs,cfg.delta[0],cfg.delta[1])
peakThetaBaseMMD = multiplePeakPower((np.transpose(psd_baseMMD,(0,2,1))),freqs,cfg.theta[0],cfg.theta[1])
peakAlphaBaseMMD = multiplePeakPower((np.transpose(psd_baseMMD,(0,2,1))),freqs,cfg.alpha[0],cfg.alpha[1])
peakBetaBaseMMD = multiplePeakPower((np.transpose(psd_baseMMD,(0,2,1))),freqs,cfg.beta[0],cfg.beta[1])
peakGammaBaseMMD = multiplePeakPower((np.transpose(psd_baseMMD,(0,2,1))),freqs,cfg.gamma[0],cfg.gamma[1])
peakDelta6MMD = multiplePeakPower((np.transpose(psd_6MMD,(0,2,1))),freqs,cfg.delta[0],cfg.delta[1])
peakTheta6MMD = multiplePeakPower((np.transpose(psd_6MMD,(0,2,1))),freqs,cfg.theta[0],cfg.theta[1])
peakAlpha6MMD = multiplePeakPower((np.transpose(psd_6MMD,(0,2,1))),freqs,cfg.alpha[0],cfg.alpha[1])
peakBeta6MMD = multiplePeakPower((np.transpose(psd_6MMD,(0,2,1))),freqs,cfg.beta[0],cfg.beta[1])
peakGamma6MMD = multiplePeakPower((np.transpose(psd_6MMD,(0,2,1))),freqs,cfg.gamma[0],cfg.gamma[1])
peakDelta12MMD = multiplePeakPower((np.transpose(psd_12MMD,(0,2,1))),freqs,cfg.delta[0],cfg.delta[1])
peakTheta12MMD = multiplePeakPower((np.transpose(psd_12MMD,(0,2,1))),freqs,cfg.theta[0],cfg.theta[1])
peakAlpha12MMD = multiplePeakPower((np.transpose(psd_12MMD,(0,2,1))),freqs,cfg.alpha[0],cfg.alpha[1])
peakBeta12MMD = multiplePeakPower((np.transpose(psd_12MMD,(0,2,1))),freqs,cfg.beta[0],cfg.beta[1])
peakGamma12MMD = multiplePeakPower((np.transpose(psd_12MMD,(0,2,1))),freqs,cfg.gamma[0],cfg.gamma[1])
 
peakDeltaBaseSD = multiplePeakPower((np.transpose(psd_baseSD,(0,2,1))),freqs,cfg.delta[0],cfg.delta[1])
peakThetaBaseSD = multiplePeakPower((np.transpose(psd_baseSD,(0,2,1))),freqs,cfg.theta[0],cfg.theta[1])
peakAlphaBaseSD = multiplePeakPower((np.transpose(psd_baseSD,(0,2,1))),freqs,cfg.alpha[0],cfg.alpha[1])
peakBetaBaseSD = multiplePeakPower((np.transpose(psd_baseSD,(0,2,1))),freqs,cfg.beta[0],cfg.beta[1])
peakGammaBaseSD = multiplePeakPower((np.transpose(psd_baseSD,(0,2,1))),freqs,cfg.gamma[0],cfg.gamma[1])
peakDelta6SD = multiplePeakPower((np.transpose(psd_6SD,(0,2,1))),freqs,cfg.delta[0],cfg.delta[1])
peakTheta6SD = multiplePeakPower((np.transpose(psd_6SD,(0,2,1))),freqs,cfg.theta[0],cfg.theta[1])
peakAlpha6SD = multiplePeakPower((np.transpose(psd_6SD,(0,2,1))),freqs,cfg.alpha[0],cfg.alpha[1])
peakBeta6SD = multiplePeakPower((np.transpose(psd_6SD,(0,2,1))),freqs,cfg.beta[0],cfg.beta[1])
peakGamma6SD = multiplePeakPower((np.transpose(psd_6SD,(0,2,1))),freqs,cfg.gamma[0],cfg.gamma[1])
peakDelta12SD = multiplePeakPower((np.transpose(psd_12SD,(0,2,1))),freqs,cfg.delta[0],cfg.delta[1])
peakTheta12SD = multiplePeakPower((np.transpose(psd_12SD,(0,2,1))),freqs,cfg.theta[0],cfg.theta[1])
peakAlpha12SD = multiplePeakPower((np.transpose(psd_12SD,(0,2,1))),freqs,cfg.alpha[0],cfg.alpha[1])
peakBeta12SD = multiplePeakPower((np.transpose(psd_12SD,(0,2,1))),freqs,cfg.beta[0],cfg.beta[1])
peakGamma12SD = multiplePeakPower((np.transpose(psd_12SD,(0,2,1))),freqs,cfg.gamma[0],cfg.gamma[1])

#%%  Generate labels
absfull_train = np.concatenate((absFullBaseND,absFullBaseMMD,absFullBaseSD))
absfull_test_6 = np.concatenate((absFull6ND,absFull6MMD,absFull6SD))
absfull_test_12 = np.concatenate((absFull12ND,absFull12MMD,absFull12SD))

relDelta_train = np.concatenate((relDeltaBaseND,relDeltaBaseMMD,relDeltaBaseSD))
relDelta_test_6 = np.concatenate((relDelta6ND,relDelta6MMD,relDelta6SD))
relDelta_test_12 = np.concatenate((relDelta12ND,relDelta12MMD,relDelta12SD))
relTheta_train = np.concatenate((relThetaBaseND,relThetaBaseMMD,relThetaBaseSD))
relTheta_test_6 = np.concatenate((relTheta6ND,relTheta6MMD,relTheta6SD))
relTheta_test_12 = np.concatenate((relTheta12ND,relTheta12MMD,relTheta12SD))
relAlpha_train = np.concatenate((relAlphaBaseND,relAlphaBaseMMD,relAlphaBaseSD))
relAlpha_test_6 = np.concatenate((relAlpha6ND,relAlpha6MMD,relAlpha6SD))
relAlpha_test_12 = np.concatenate((relAlpha12ND,relAlpha12MMD,relAlpha12SD))
relBeta_train = np.concatenate((relBetaBaseND,relBetaBaseMMD,relBetaBaseSD))
relBeta_test_6 = np.concatenate((relBeta6ND,relBeta6MMD,relBeta6SD))
relBeta_test_12 = np.concatenate((relBeta12ND,relBeta12MMD,relBeta12SD))
relGamma_train = np.concatenate((relGammaBaseND,relGammaBaseMMD,relGammaBaseSD))
relGamma_test_6 = np.concatenate((relGamma6ND,relGamma6MMD,relGamma6SD))
relGamma_test_12 = np.concatenate((relGamma12ND,relGamma12MMD,relGamma12SD))

peakDelta_train = np.concatenate((peakDeltaBaseND,peakDeltaBaseMMD,peakDeltaBaseSD))
peakDelta_test_6 = np.concatenate((peakDelta6ND,peakDelta6MMD,peakDelta6SD))
peakDelta_test_12 = np.concatenate((peakDelta12ND,peakDelta12MMD,peakDelta12SD))
peakTheta_train = np.concatenate((peakThetaBaseND,peakThetaBaseMMD,peakThetaBaseSD))
peakTheta_test_6 = np.concatenate((peakTheta6ND,peakTheta6MMD,peakTheta6SD))
peakTheta_test_12 = np.concatenate((peakTheta12ND,peakTheta12MMD,peakTheta12SD))
peakAlpha_train = np.concatenate((peakAlphaBaseND,peakAlphaBaseMMD,peakAlphaBaseSD))
peakAlpha_test_6 = np.concatenate((peakAlpha6ND,peakAlpha6MMD,peakAlpha6SD))
peakAlpha_test_12 = np.concatenate((peakAlpha12ND,peakAlpha12MMD,peakAlpha12SD))
peakBeta_train = np.concatenate((peakBetaBaseND,peakBetaBaseMMD,peakBetaBaseSD))
peakBeta_test_6 = np.concatenate((peakBeta6ND,peakBeta6MMD,peakBeta6SD))
peakBeta_test_12 = np.concatenate((peakBeta12ND,peakBeta12MMD,peakBeta12SD))
peakGamma_train = np.concatenate((peakGammaBaseND,peakGammaBaseMMD,peakGammaBaseSD))
peakGamma_test_6 = np.concatenate((peakGamma6ND,peakGamma6MMD,peakGamma6SD))
peakGamma_test_12 = np.concatenate((peakGamma12ND,peakGamma12MMD,peakGamma12SD))

labels_ND_train,labels_MMD_train,labels_SD_train = np.repeat('ND',len(relDeltaBaseND)),np.repeat('MMD',len(relDeltaBaseMMD)),np.repeat('SD',len(relDeltaBaseSD))
classes_train = np.concatenate((labels_ND_train,labels_MMD_train,labels_SD_train))
filenames_6 = np.concatenate((sixND,sixMMD,sixSD))
filenames_12 = np.concatenate((twelveND,twelveMMD,twelveSD))
train_df = pd.DataFrame({'fz_AbsGammaPower':absfull_train[:,0],'cz_AbsGammaPower':absfull_train[:,1],'pz_AbsGammaPower':absfull_train[:,2],
                         'fz_RelativeDeltaPower':relDelta_train[:,0],'cz_RelativeDeltaPower':relDelta_train[:,1],'pz_RelativeDeltaPower':relDelta_train[:,2],
                         'fz_RelativeThetaPower':relTheta_train[:,0],'cz_RelativeThetaPower':relTheta_train[:,1],'pz_RelativeThetaPower':relTheta_train[:,2],
                            'fz_RelativeAlphaPower':relAlpha_train[:,0],'cz_RelativeAlphaPower':relAlpha_train[:,1],'pz_RelativeAlphaPower':relAlpha_train[:,2],
                            'fz_RelativeBetaPower':relBeta_train[:,0],'cz_RelativeBetaPower':relBeta_train[:,1],'pz_RelativeBetaPower':relBeta_train[:,2],
                            'fz_RelativeGammaPower':relGamma_train[:,0],'cz_RelativeGammaPower':relGamma_train[:,1],'pz_RelativeGammaPower':relGamma_train[:,2],
                            'fz_PeakDeltaPower':peakDelta_train[:,0],'cz_PeakDeltaPower':peakDelta_train[:,1],'pz_PeakDeltaPower':peakDelta_train[:,2],
                            'fz_PeakThetaPower':peakTheta_train[:,0],'cz_PeakThetaPower':peakTheta_train[:,1],'pz_PeakThetaPower':peakTheta_train[:,2],
                            'fz_PeakAlphaPower':peakAlpha_train[:,0],'cz_PeakAlphaPower':peakAlpha_train[:,1],'pz_PeakAlphaPower':peakAlpha_train[:,2],
                            'fz_PeakBetaPower':peakBeta_train[:,0],'cz_PeakBetaPower':peakBeta_train[:,1],'pz_PeakBetaPower':peakBeta_train[:,2],
                            'fz_PeakGammaPower':peakGamma_train[:,0],'cz_PeakGammaPower':peakGamma_train[:,1],'pz_PeakGammaPower':peakGamma_train[:,2],
                            'Class':classes_train})
test_6_df = pd.DataFrame({'filenames':filenames_6,'fz_AbsGammaPower':absfull_test_6[:,0],'cz_AbsGammaPower':absfull_test_6[:,1],'pz_AbsGammaPower':absfull_test_6[:,2],
                            'fz_RelativeDeltaPower':relDelta_test_6[:,0],'cz_RelativeDeltaPower':relDelta_test_6[:,1],'pz_RelativeDeltaPower':relDelta_test_6[:,2],
                            'fz_RelativeThetaPower':relTheta_test_6[:,0],'cz_RelativeThetaPower':relTheta_test_6[:,1],'pz_RelativeThetaPower':relTheta_test_6[:,2],
                            'fz_RelativeAlphaPower':relAlpha_test_6[:,0],'cz_RelativeAlphaPower':relAlpha_test_6[:,1],'pz_RelativeAlphaPower':relAlpha_test_6[:,2],
                            'fz_RelativeBetaPower':relBeta_test_6[:,0],'cz_RelativeBetaPower':relBeta_test_6[:,1],'pz_RelativeBetaPower':relBeta_test_6[:,2],
                            'fz_RelativeGammaPower':relGamma_test_6[:,0],'cz_RelativeGammaPower':relGamma_test_6[:,1],'pz_RelativeGammaPower':relGamma_test_6[:,2],
                            'fz_PeakDeltaPower':peakDelta_test_6[:,0],'cz_PeakDeltaPower':peakDelta_test_6[:,1],'pz_PeakDeltaPower':peakDelta_test_6[:,2],
                            'fz_PeakThetaPower':peakTheta_test_6[:,0],'cz_PeakThetaPower':peakTheta_test_6[:,1],'pz_PeakThetaPower':peakTheta_test_6[:,2],
                            'fz_PeakAlphaPower':peakAlpha_test_6[:,0],'cz_PeakAlphaPower':peakAlpha_test_6[:,1],'pz_PeakAlphaPower':peakAlpha_test_6[:,2],
                            'fz_PeakBetaPower':peakBeta_test_6[:,0],'cz_PeakBetaPower':peakBeta_test_6[:,1],'pz_PeakBetaPower':peakBeta_test_6[:,2],
                            'fz_PeakGammaPower':peakGamma_test_6[:,0],'cz_PeakGammaPower':peakGamma_test_6[:,1],'pz_PeakGammaPower':peakGamma_test_6[:,2]})
test_12_df = pd.DataFrame({'filenames':filenames_12,'fz_AbsGammaPower':absfull_test_12[:,0],'cz_AbsGammaPower':absfull_test_12[:,1],'pz_AbsGammaPower':absfull_test_12[:,2],
                            'fz_RelativeDeltaPower':relDelta_test_12[:,0],'cz_RelativeDeltaPower':relDelta_test_12[:,1],'pz_RelativeDeltaPower':relDelta_test_12[:,2],
                            'fz_RelativeThetaPower':relTheta_test_12[:,0],'cz_RelativeThetaPower':relTheta_test_12[:,1],'pz_RelativeThetaPower':relTheta_test_12[:,2],
                            'fz_RelativeAlphaPower':relAlpha_test_12[:,0],'cz_RelativeAlphaPower':relAlpha_test_12[:,1],'pz_RelativeAlphaPower':relAlpha_test_12[:,2],
                            'fz_RelativeBetaPower':relBeta_test_12[:,0],'cz_RelativeBetaPower':relBeta_test_12[:,1],'pz_RelativeBetaPower':relBeta_test_12[:,2],
                            'fz_RelativeGammaPower':relGamma_test_12[:,0],'cz_RelativeGammaPower':relGamma_test_12[:,1],'pz_RelativeGammaPower':relGamma_test_12[:,2],
                            'fz_PeakDeltaPower':peakDelta_test_12[:,0],'cz_PeakDeltaPower':peakDelta_test_12[:,1],'pz_PeakDeltaPower':peakDelta_test_12[:,2],
                            'fz_PeakThetaPower':peakTheta_test_12[:,0],'cz_PeakThetaPower':peakTheta_test_12[:,1],'pz_PeakThetaPower':peakTheta_test_12[:,2],
                            'fz_PeakAlphaPower':peakAlpha_test_12[:,0],'cz_PeakAlphaPower':peakAlpha_test_12[:,1],'pz_PeakAlphaPower':peakAlpha_test_12[:,2],
                            'fz_PeakBetaPower':peakBeta_test_12[:,0],'cz_PeakBetaPower':peakBeta_test_12[:,1],'pz_PeakBetaPower':peakBeta_test_12[:,2],
                            'fz_PeakGammaPower':peakGamma_test_12[:,0],'cz_PeakGammaPower':peakGamma_test_12[:,1],'pz_PeakGammaPower':peakGamma_test_12[:,2]})

df_dir = '/Users/joshuaighalo/Documents/GitHub/eegDementia/MLOps/dementiaStages/dataframes/'
train_df.to_csv(df_dir + 'train_10.csv', index=False)
test_6_df.to_csv(df_dir + 'test_6m_10.csv', index=False)
test_12_df.to_csv(df_dir + 'test_12m_10.csv', index=False)

