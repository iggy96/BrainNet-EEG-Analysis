from cf_libs import *

path = "/Users/joshuaighalo/Documents/BrainNet/Projects/Workspace/Datasets/cognitiveFatigue/new/"

psdBND_1,psdBND_2 = np.load(path + "psdBND_1.npy"),np.load(path + "psdBND_2.npy")
psd6ND_1,psd6ND_2 = np.load(path + "psd6ND_1.npy"),np.load(path + "psd6ND_2.npy")
psd12ND_1,psd12ND_2 = np.load(path + "psd12ND_1.npy"),np.load(path + "psd12ND_2.npy")
psdBMMD_1,psdBMMD_2 = np.load(path + "psdBMMD_1.npy"),np.load(path + "psdBMMD_2.npy")
psd6MMD_1,psd6MMD_2 = np.load(path + "psd6MMD_1.npy"),np.load(path + "psd6MMD_2.npy")
psd12MMD_1,psd12MMD_2 = np.load(path + "psd12MMD_1.npy"),np.load(path + "psd12MMD_2.npy")
psdBSD_1,psdBSD_2 = np.load(path + "psdBSD_1.npy"),np.load(path + "psdBSD_2.npy")
psd6SD_1,psd6SD_2 = np.load(path + "psd6SD_1.npy"),np.load(path + "psd6SD_2.npy")
psd12SD_1,psd12SD_2 = np.load(path + "psd12SD_1.npy"),np.load(path + "psd12SD_2.npy")
freqs = psd12SD_1[4][0]




def featuresGenerator(powerSpectralDensities_1,powerSpectralDensities_2,frequency):
    def peakPower(psd_1,psd_2,freqs,frqrange):
        #   Utilizes MultiTaper method to calculate the average power of a band
        #  Inputs  :   data    - 2D numpy array (d0 = samples, d1 = channels) of filtered EEG data
        #              low     - lower limit in Hz for the brain wave
        #              high    - upper limit in Hz for the brain wave
        #   Output  :   3D array (columns of array,no of windows,window size)
        def param_1(data_1,frequency,freqrange):
            idxFreq_1 = np.abs(frequency - freqrange[0]).argmin()
            idxFreq_2 = np.abs(frequency - freqrange[1]).argmin()
            psdRange = data_1[idxFreq_1:idxFreq_2]
            peakPower = np.max(psdRange)
            return peakPower  
        def param_2(data_2,frequency,freqrange):
            idxFreq_1 = np.abs(frequency - freqrange[0]).argmin()
            idxFreq_2 = np.abs(frequency - freqrange[1]).argmin()
            psdRange = data_2[idxFreq_1:idxFreq_2]
            peakPower = np.max(psdRange)
            return peakPower

        peakPower_1,peakPower_2 = [],[]
        for i in range(len(psd_1)):
            peakPower_1.append(param_1(psd_1[i],freqs,frqrange))
        peakPower_1 = np.array(peakPower_1)
        for i in range(len(psd_2)):
            peakPower_2.append(param_2(psd_2[i],freqs,frqrange))
        peakPower_2 = np.array(peakPower_2)
        return peakPower_1,peakPower_2 
    
    #   Compute the peak power in the alpha band
    fz_alpha_peakPower_1,fz_alpha_peakPower_2 = peakPower(psd_1=powerSpectralDensities_1[0],psd_2=powerSpectralDensities_2[0],freqs=frequency,frqrange=[8,12])
    cz_alpha_peakPower_1,cz_alpha_peakPower_2 = peakPower(psd_1=powerSpectralDensities_1[1],psd_2=powerSpectralDensities_2[1],freqs=frequency,frqrange=[8,12])
    pz_alpha_peakPower_1,pz_alpha_peakPower_2 = peakPower(psd_1=powerSpectralDensities_1[2],psd_2=powerSpectralDensities_2[2],freqs=frequency,frqrange=[8,12])
    global_alpha_peakPower_1,global_alpha_peakPower_2 = peakPower(psd_1=powerSpectralDensities_1[3],psd_2=powerSpectralDensities_2[3],freqs=frequency,frqrange=[8,12])
    #   Compute the difference between the peak power in the alpha band
    fz_alpha_diff_1,fz_alpha_diff_2 = fz_alpha_peakPower_1 - fz_alpha_peakPower_2,fz_alpha_peakPower_2 - fz_alpha_peakPower_1
    cz_alpha_diff_1,cz_alpha_diff_2 = cz_alpha_peakPower_1 - cz_alpha_peakPower_2,cz_alpha_peakPower_2 - cz_alpha_peakPower_1
    pz_alpha_diff_1,pz_alpha_diff_2 = pz_alpha_peakPower_1 - pz_alpha_peakPower_2,pz_alpha_peakPower_2 - pz_alpha_peakPower_1
    global_alpha_diff_1,global_alpha_diff_2 = global_alpha_peakPower_1 - global_alpha_peakPower_2,global_alpha_peakPower_2 - global_alpha_peakPower_1
    #   Compute ratio between the peak power in the alpha band
    fz_alpha_ratio_1,fz_alpha_ratio_2 = fz_alpha_peakPower_1/fz_alpha_peakPower_2,fz_alpha_peakPower_2/fz_alpha_peakPower_1
    cz_alpha_ratio_1,cz_alpha_ratio_2 = cz_alpha_peakPower_1/cz_alpha_peakPower_2,cz_alpha_peakPower_2/cz_alpha_peakPower_1
    pz_alpha_ratio_1,pz_alpha_ratio_2 = pz_alpha_peakPower_1/pz_alpha_peakPower_2,pz_alpha_peakPower_2/pz_alpha_peakPower_1
    global_alpha_ratio_1,global_alpha_ratio_2 = global_alpha_peakPower_1/global_alpha_peakPower_2,global_alpha_peakPower_2/global_alpha_peakPower_1
    #   Extract labels
    labels_1 = np.repeat(0,len(fz_alpha_peakPower_1))
    labels_2 = np.repeat(1,len(fz_alpha_peakPower_2))
    #   Gather features for run 1
    features_1 = np.vstack((fz_alpha_peakPower_1,cz_alpha_peakPower_1,pz_alpha_peakPower_1,global_alpha_peakPower_1,
                            fz_alpha_diff_1,cz_alpha_diff_1,pz_alpha_diff_1,global_alpha_diff_1,
                            fz_alpha_ratio_1,cz_alpha_ratio_1,pz_alpha_ratio_1,global_alpha_ratio_1,
                            labels_1)).T
    #   Gather features for run 2
    features_2 = np.vstack((fz_alpha_peakPower_2,cz_alpha_peakPower_2,pz_alpha_peakPower_2,global_alpha_peakPower_2,
                            fz_alpha_diff_2,cz_alpha_diff_2,pz_alpha_diff_2,global_alpha_diff_2,
                            fz_alpha_ratio_2,cz_alpha_ratio_2,pz_alpha_ratio_2,global_alpha_ratio_2,
                            labels_2)).T

    features = np.vstack((features_1,features_2))
    
    df = pd.DataFrame({'fz_alpha_peakPower':features[:,0],'cz_alpha_peakPower':features[:,1],'pz_alpha_peakPower':features[:,2],'global_alpha_peakPower':features[:,3],
                          'fz_alpha_diff':features[:,4],'cz_alpha_diff':features[:,5],'pz_alpha_diff':features[:,6],'global_alpha_diff':features[:,7],
                            'fz_alpha_ratio':features[:,8],'cz_alpha_ratio':features[:,9],'pz_alpha_ratio':features[:,10],'global_alpha_ratio':features[:,11],
                            'labels':features[:,12]})
    df = df.sample(frac=1).reset_index()
    print("Features extracted")
    return df
    

featuresBND,features6ND,features12ND = featuresGenerator(psdBND_1,psdBND_2,freqs),featuresGenerator(psd6ND_1,psd6ND_2,freqs),featuresGenerator(psd12ND_1,psd12ND_2,freqs)
featuresBMMD,features6MMD,features12MMD = featuresGenerator(psdBMMD_1,psdBMMD_2,freqs),featuresGenerator(psd6MMD_1,psd6MMD_2,freqs),featuresGenerator(psd12MMD_1,psd12MMD_2,freqs)
featuresBSD,features6SD,features12SD = featuresGenerator(psdBSD_1,psdBSD_2,freqs),featuresGenerator(psd6SD_1,psd6SD_2,freqs),featuresGenerator(psd12SD_1,psd12SD_2,freqs)




