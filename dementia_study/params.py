import numpy as np

# Analysis parameters
figure_size = (6,16)
plot_color = ['#1f77b4', '#ff7f0e', '#2ca02c']
channelNames = ['Fz','Cz','Pz']
plot_color_1 = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
win_size = 2560
step_size = 1280
nfft = 500
noverlap = 256


# EEG parameters
epoch = 1000  # ms - total window size
prestim = 100  # ms - duration  of the baseline (pre-stimulus interval) within the epoch
lowPass = 20  # Hz - cut-off for the low-pass filter
highPass = 0.3  # Hz - cut-off for the high-pass filter
line = 60  # Hz - line voltage
clip = 75  # uV - clipping threshold
fs = 500  # todo - move this to the hardware config
stimTrig = dict(std=[1], dev=[2], con=[4, 7], inc=[5, 8])
stimTypes = ['std', 'dev', 'con', 'inc'] 
stimNum = dict(std = [264], dev = [24], con = [36], inc = [36])

erpTypes = ['N100', 'P300', 'N400']
erpStim = dict(N100 = ['std', 'dev'], P300 = ['std', 'dev'], N400 = ['con', 'inc'])
erpRange = dict(N100 = [70, 200], P300 = [200, 500], N400 = [300, 800])

ant = dict(
    fs=500,                            # Hz - sampling rate of the amplifier
    dataCols=12,                       # Number of columns in the EEG
    nEEG=8,                            # Number of EEG Channels
    trigCol=-1,                        # Position of Trigger Channel
    uvScale=1000000,                   # antNeuro displays numbers in volts
    dt=np.float64,                     # data type to read in
    eegChans=np.array([0, 1, 2]),      # index of Fz, Cz, Pz within raw data array
    eegChanNames = ['Fz', 'Cz', 'Pz'],
    eogChans = np.array([5]),  # FPZ for now    
    afSS=0                             # Start sample of the adaptive filter (transient)
)


gtec = dict(
    fs=500,                            # Hz - sampling rate of the amplifier
    dataCols=14,                       # Number of columns in the EEG
    nEEG=8,                            # Number of EEG Channels
    trigCol=-1,                        # Position of Trigger Channel
    uvScale=1,                         # gtec already displays numbers in microvolts
    dt=np.float32,                     # data type to read in
    eegChans=np.array([0, 1, 3]),      # index of Fz, Cz, Pz within raw data array
    eegChanNames=['Fz', 'Cz', 'Pz'],
    eogChans=np.array([2,4,5,6,7]),    # index of EOG (P3 P4 P07 P08 OZ) channels within raw data 
    afSS=3000                          # Start sample of the adaptive filter (transient)
)
