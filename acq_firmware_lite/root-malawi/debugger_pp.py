from fn_libs import *

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

def extract_epochs(stim,data, trigger_value, pre_time, post_time, sampling_rate, clip_value):
    
    def baseline_correction(epoch, pre_samples):
        baseline = np.nanmean(epoch[:pre_samples])
        return epoch - baseline

    def artifact_rejection(epoch, clip_value):
        return np.ptp(epoch) <= clip_value

    if stim == 'Tones':
        # Calculate number of samples for pre_time and post_time
        pre_samples = pre_time * sampling_rate // 1000
        post_samples = post_time * sampling_rate // 1000

        # Find the indices where the trigger column is equal to the trigger value
        trigger_indices = np.where(data[:, 1] == trigger_value)[0]

        eeg_epochs = []

        for index in trigger_indices:
            # Ensure index isn't too early in the data array
            if index - pre_samples >= 0:
                # Get data from pre_samples before to post_samples after trigger
                epoch = data[index - pre_samples: index + post_samples, 0]
                # Ensure epoch is the correct length (some triggers might be too close to the end of the data)
                if len(epoch) == pre_samples + post_samples:
                    # Perform baseline correction
                    epoch = baseline_correction(epoch, pre_samples)
                    # Perform artifact rejection
                    if artifact_rejection(epoch, clip_value):
                        eeg_epochs.append(epoch)

        # Convert list of epochs to a 3D numpy array
        eeg_epochs_array = np.array(eeg_epochs)
       # print("Tones eeg_epochs_array.shape:", eeg_epochs_array.shape)
        if eeg_epochs_array.shape[0] == 0:
            eeg_epochs_array = np.empty((2, 500))
            eeg_epochs_array[:] = np.nan
        else:
            pass

        # Calculate ERP
        print(eeg_epochs_array.shape)
        try:
            with warnings.catch_warnings(record=True) as w:  # Context manager to catch warnings
                warnings.simplefilter("always")  # Always trigger warnings to catch them
                erp = np.nanmean(eeg_epochs_array, axis=0)
                
                if w and issubclass(w[-1].category, RuntimeWarning):  # Check if a RuntimeWarning occurred
                    raise RuntimeWarning("Mean of empty slice detected.")  # Raise it as an exception
        except (TypeError, RuntimeWarning):  # Catch both TypeError and RuntimeWarning
            erp = np.zeros(500)  # Default value of shape (500,)
            print("An error or warning occurred. Using default value for erp.")


    elif stim == 'Words':
        # Calculate number of samples for pre_time and post_time
        pre_samples = pre_time * sampling_rate // 1000
        post_samples = post_time * sampling_rate // 1000

        # Find the indices where the trigger column is equal to the trigger value
        trigger_indices = np.where(data[:, 1] == trigger_value)[0]

        eeg_epochs = []

        for index in trigger_indices:
            # Ensure index isn't too early in the data array
            if index - pre_samples >= 0:
                # Get data from pre_samples before to post_samples after trigger
                epoch = data[index - pre_samples: index + post_samples, 0]
                # Ensure epoch is the correct length (some triggers might be too close to the end of the data)
                if len(epoch) == pre_samples + post_samples:
                    # Perform baseline correction
                    epoch = baseline_correction(epoch, pre_samples)
                    # Perform artifact rejection
                    if artifact_rejection(epoch, clip_value):
                        eeg_epochs.append(epoch)

        # Convert list of epochs to a 3D numpy array
        eeg_epochs_array = np.array(eeg_epochs)
       # print("Tones eeg_epochs_array.shape:", eeg_epochs_array.shape)
        if eeg_epochs_array.shape[0] == 0:
            eeg_epochs_array = np.empty((2, 500))
            eeg_epochs_array[:] = np.nan
        else:
            pass

        # Calculate ERP
        print(eeg_epochs_array.shape)
        try:
            with warnings.catch_warnings(record=True) as w:  # Context manager to catch warnings
                warnings.simplefilter("always")  # Always trigger warnings to catch them
                erp = np.nanmean(eeg_epochs_array, axis=0)
                
                if w and issubclass(w[-1].category, RuntimeWarning):  # Check if a RuntimeWarning occurred
                    raise RuntimeWarning("Mean of empty slice detected.")  # Raise it as an exception
        except (TypeError, RuntimeWarning):  # Catch both TypeError and RuntimeWarning
            erp = np.zeros(500)  # Default value of shape (500,)
            print("An error or warning occurred. Using default value for erp.")

    return erp

def rising_edge(samples_data,trigger_vals):
    def handle_repeated_triggers(data,idx):
        idx_diff = [0] + [idx[i+1] - idx[i] for i in range(len(idx)-1)]
        idx_1 = [i for i, e in enumerate(idx_diff) if e == 1]
        idx_repeat = [idx[i] for i in idx_1]
        for i in idx_repeat:
            data[i] = 0
        data
        return data

    # Check for each trigger value
    last_column = samples_data[:, -1]


    indices_1 = np.where(last_column == 1)[0]
    indices_2 = np.where(last_column == 2)[0]
    indices_4 = np.where(last_column == 4)[0]
    indices_5 = np.where(last_column == 5)[0]

    last_column = handle_repeated_triggers(last_column,indices_1)
    last_column = handle_repeated_triggers(last_column,indices_2)
    last_column = handle_repeated_triggers(last_column,indices_4)
    last_column = handle_repeated_triggers(last_column,indices_5)

   # print('POST PROCESS TRIGGERS')
    unique_trigger_vals = np.unique(trigger_vals)
    for trigger_value in unique_trigger_vals:
        indices = np.where(last_column == trigger_value)[0]
        count = len(indices)
      #  print(f"Trigger value {trigger_value} appears {count} times in the last column")
      #  print(f"Indices where {trigger_value} appears: {indices}")

    samples_data[:, -1] = last_column 
    return samples_data

def post_process_onebatch(dataAcquired,fs,clip,line,low,high,trigger_vals):
    dataAcquired = rising_edge(dataAcquired,trigger_vals)
    eegData, eogData = dataAcquired[:,[0,1,3]],dataAcquired[:,[5,7]]
    filtering = filters()
    afEEG = filtering.adaptive(eegData,eogData)
    nfEEG = filtering.notch(afEEG,line,fs)
    bpfEEG = filtering.butterBandPass(nfEEG,low,high,fs)
    bpfEEG = np.column_stack((bpfEEG, dataAcquired[:, 8]))

    std,dev,con,inc = 1,2,4,5
    erp_std_fz = extract_epochs('Tones',bpfEEG[:,[0,3]], trigger_value=std, pre_time=100, post_time=900, sampling_rate=500, clip_value=clip)
    erp_dev_fz = extract_epochs('Tones',bpfEEG[:,[0,3]], trigger_value=dev, pre_time=100, post_time=900, sampling_rate=500, clip_value=clip)
    erp_std_cz = extract_epochs('Tones',bpfEEG[:,[1,3]], trigger_value=std, pre_time=100, post_time=900, sampling_rate=500, clip_value=clip)
    erp_dev_cz = extract_epochs('Tones',bpfEEG[:,[1,3]], trigger_value=dev, pre_time=100, post_time=900, sampling_rate=500, clip_value=clip)
    erp_std_pz = extract_epochs('Tones',bpfEEG[:,[2,3]], trigger_value=std, pre_time=100, post_time=900, sampling_rate=500, clip_value=clip)
    erp_dev_pz = extract_epochs('Tones',bpfEEG[:,[2,3]], trigger_value=dev, pre_time=100, post_time=900, sampling_rate=500, clip_value=clip)
    erp_con_fz = extract_epochs('Words',bpfEEG[:,[0,3]], trigger_value=con, pre_time=100, post_time=900, sampling_rate=500, clip_value=clip)
    erp_inc_fz = extract_epochs('Words',bpfEEG[:,[0,3]], trigger_value=inc, pre_time=100, post_time=900, sampling_rate=500, clip_value=clip)
    erp_con_cz = extract_epochs('Words',bpfEEG[:,[1,3]], trigger_value=con, pre_time=100, post_time=900, sampling_rate=500, clip_value=clip)
    erp_inc_cz = extract_epochs('Words',bpfEEG[:,[1,3]], trigger_value=inc, pre_time=100, post_time=900, sampling_rate=500, clip_value=clip)
    erp_con_pz = extract_epochs('Words',bpfEEG[:,[2,3]], trigger_value=con, pre_time=100, post_time=900, sampling_rate=500, clip_value=clip)
    erp_inc_pz = extract_epochs('Words',bpfEEG[:,[2,3]], trigger_value=inc, pre_time=100, post_time=900, sampling_rate=500, clip_value=clip)
    latency = np.arange(-100,900,2)
    return latency,erp_std_fz,erp_dev_fz,erp_std_cz,erp_dev_cz,erp_std_pz,erp_dev_pz,erp_con_fz,erp_inc_fz,erp_con_cz,erp_inc_cz,erp_con_pz,erp_inc_pz

def generate_report_pdf(latency,
                        erp_std_fz,erp_dev_fz,
                        erp_std_cz,erp_dev_cz,
                        erp_std_pz,erp_dev_pz,
                        erp_con_fz,erp_inc_fz,
                        erp_con_cz,erp_inc_cz,
                        erp_con_pz,erp_inc_pz):
    
    with open(r"..\assets\utils\biodata.txt", 'r') as f:
        lines = [next(f) for _ in range(13)]

    #   Figure
    fig, axes = plt.subplots(nrows=2, ncols=3, sharex=False, sharey=True, figsize=(15,10))
    # add a header at the top center of the figure
    fig.suptitle('Scan Result (Preview)',x=0.7, fontsize=50, fontweight='bold',va='top',ha='left')
    # add subheading below the header
    fig.text(0.7, 0.83, 'ERP Waveforms', fontsize=30, fontweight='bold',va='top',ha='right')

    label_1,label_2 = ['Standard', 'Deviant'], ['Congruent', 'Incongruent']
    line_thickness,shade_thickness,tick_size,legend_size = 2.5,0.1,9,12
    sd_erp_std_fz, sd_erp_dev_fz, sd_erp_std_cz, sd_erp_dev_cz, sd_erp_std_pz, sd_erp_dev_pz = np.nanstd(erp_std_fz, axis=0), np.nanstd(erp_dev_fz, axis=0), np.nanstd(erp_std_cz, axis=0), np.nanstd(erp_dev_cz, axis=0), np.nanstd(erp_std_pz, axis=0), np.nanstd(erp_dev_pz, axis=0)
    sd_erp_con_fz, sd_erp_inc_fz, sd_erp_con_cz, sd_erp_inc_cz, sd_erp_con_pz, sd_erp_inc_pz = np.nanstd(erp_con_fz, axis=0), np.nanstd(erp_inc_fz, axis=0), np.nanstd(erp_con_cz, axis=0), np.nanstd(erp_inc_cz, axis=0), np.nanstd(erp_con_pz, axis=0), np.nanstd(erp_inc_pz, axis=0)

    axes[0, 0].plot(latency, erp_std_fz, color='blue', linewidth=line_thickness)
    axes[0, 0].plot(latency, erp_dev_fz, color='red', linewidth=line_thickness)
    axes[0, 0].fill_between(latency, erp_std_fz - sd_erp_std_fz, erp_std_fz + sd_erp_std_fz, alpha=shade_thickness, color='blue')
    axes[0, 0].fill_between(latency, erp_dev_fz - sd_erp_dev_fz, erp_dev_fz + sd_erp_dev_fz, alpha=shade_thickness, color='red')
    axes[0, 0].axvline(0, color='black', linestyle='--')
    axes[0, 0].invert_yaxis()
    axes[0, 0].set_title('Fz', fontsize=20)
    axes[0, 0].tick_params(axis='both', which='major', labelsize=tick_size)
    axes[0, 0].xaxis.set_major_locator(ticker.MultipleLocator(100))
    axes[0, 0].set_xlim(-100, 900)

    axes[0, 1].plot(latency, erp_std_cz, color='blue', linewidth=line_thickness)
    axes[0, 1].plot(latency, erp_dev_cz, color='red', linewidth=line_thickness)
    axes[0, 1].fill_between(latency, erp_std_cz - sd_erp_std_cz, erp_std_cz + sd_erp_std_cz, alpha=shade_thickness, color='blue')
    axes[0, 1].fill_between(latency, erp_dev_cz - sd_erp_dev_cz, erp_dev_cz + sd_erp_dev_cz, alpha=shade_thickness, color='red')
    axes[0, 1].axvline(0, color='black', linestyle='--')
    axes[0, 1].set_title('Cz', fontsize=20)
    axes[0, 1].tick_params(axis='both', which='major', labelsize=tick_size)
    axes[0, 1].xaxis.set_major_locator(ticker.MultipleLocator(100))
    axes[0, 1].set_xlim(-100, 900)


    axes[0, 2].plot(latency, erp_std_pz, color='blue', linewidth=line_thickness)
    axes[0, 2].plot(latency, erp_dev_pz, color='red', linewidth=line_thickness)
    axes[0, 2].fill_between(latency, erp_std_pz - sd_erp_std_pz, erp_std_pz + sd_erp_std_pz, alpha=shade_thickness, color='blue')
    axes[0, 2].fill_between(latency, erp_dev_pz - sd_erp_dev_pz, erp_dev_pz + sd_erp_dev_pz, alpha=shade_thickness, color='red')
    axes[0, 2].axvline(0, color='black', linestyle='--')
    axes[0, 2].set_title('Pz', fontsize=20)
    axes[0, 2].tick_params(axis='both', which='major', labelsize=tick_size)
    axes[0, 2].legend(label_1, loc='upper right', fontsize=legend_size)
    axes[0, 2].xaxis.set_major_locator(ticker.MultipleLocator(100))
    axes[0, 2].set_xlim(-100, 900)

    axes[1, 0].plot(latency, erp_con_fz, color='blue', linewidth=line_thickness)
    axes[1, 0].plot(latency, erp_inc_fz, color='red', linewidth=line_thickness)
    axes[1, 0].fill_between(latency, erp_con_fz - sd_erp_con_fz, erp_con_fz + sd_erp_con_fz, alpha=shade_thickness, color='blue')
    axes[1, 0].fill_between(latency, erp_inc_fz - sd_erp_inc_fz, erp_inc_fz + sd_erp_inc_fz, alpha=shade_thickness, color='red')
    axes[1, 0].axvline(0, color='black', linestyle='--')
    axes[1, 0].tick_params(axis='both', which='major', labelsize=tick_size)
    axes[1, 0].xaxis.set_major_locator(ticker.MultipleLocator(100))
    axes[1, 0].set_xlim(-100, 900)

    axes[1, 1].plot(latency, erp_con_cz, color='blue', linewidth=line_thickness)
    axes[1, 1].plot(latency, erp_inc_cz, color='red', linewidth=line_thickness)
    axes[1, 1].fill_between(latency, erp_con_cz - sd_erp_con_cz, erp_con_cz + sd_erp_con_cz, alpha=shade_thickness, color='blue')
    axes[1, 1].fill_between(latency, erp_inc_cz - sd_erp_inc_cz, erp_inc_cz + sd_erp_inc_cz, alpha=shade_thickness, color='red')
    axes[1, 1].axvline(0, color='black', linestyle='--')
    axes[1, 1].tick_params(axis='both', which='major', labelsize=tick_size)
    axes[1, 1].xaxis.set_major_locator(ticker.MultipleLocator(100))
    axes[1, 1].set_xlim(-100, 900)

    axes[1, 2].plot(latency, erp_con_pz, color='blue', linewidth=line_thickness)
    axes[1, 2].plot(latency, erp_inc_pz, color='red', linewidth=line_thickness)
    axes[1, 2].fill_between(latency, erp_con_pz - sd_erp_con_pz, erp_con_pz + sd_erp_con_pz, alpha=shade_thickness, color='blue')
    axes[1, 2].fill_between(latency, erp_inc_pz - sd_erp_inc_pz, erp_inc_pz + sd_erp_inc_pz, alpha=shade_thickness, color='red')
    axes[1, 2].axvline(0, color='black', linestyle='--')
    axes[1, 2].tick_params(axis='both', which='major', labelsize=tick_size)
    axes[1, 2].legend(label_2, loc='upper right', fontsize=legend_size)
    axes[1, 2].xaxis.set_major_locator(ticker.MultipleLocator(100))
    axes[1, 2].set_xlim(-100, 900)

    fig.text(0.5, 0.15, 'Latency (ms)', ha='center', va='center',fontsize=16)
    fig.text(0.09, 0.5, 'Amplitude (uV)', ha='center', va='center', rotation='vertical',fontsize=16)
    fig.subplots_adjust(top=0.75, bottom = 0.2)

    # Adding title for text lines "Biodata"
    plt.text(1.25, 0.8, 'BioData', transform=plt.gcf().transFigure, fontsize=30, fontweight='bold')

    # Add the text from the file to the plot
    for i, line in enumerate(lines):
        plt.text(1.1, 0.7 - i*0.03, line, transform=plt.gcf().transFigure, fontsize=25)

    # Extract ID from file
    with open(r"..\assets\utils\biodata.txt", "r") as file:
        lines = [line.strip() for line in file.readlines()]
        sid = lines[1].replace("ID:","").strip()
        sdate = lines[4].replace("Date:","").strip()

    # Form filename
    filename = f"{sid}_{sdate}.pdf"

    # Save the figure to a PDF
    fig.savefig(rf'..\..\user_access\reports\{filename}', format='pdf', bbox_inches='tight')
    try:
        fig.savefig(rf'..\..\user_access\reports\{filename}', format='pdf', bbox_inches='tight')
    except Exception as e:
        fig.savefig(rf'..\..\user_access\reports\{filename.replace(".pdf", ".png")}', format='png', bbox_inches='tight')

def postprocess_generate_report_1():
    with open(r"..\assets\utils\biodata.txt", "r") as file:
        lines = [line.strip() for line in file.readlines()]
 
    SID = lines[1].replace("ID:","")
    SDATE = lines[4].replace("Date:","")
    location = lines[12].replace("Location:","")
    if location == "America":
        line = 60
    else:
        line = 50
   # print("line: ",line)
    low,high = 0.1,10
    fs = 500
    clip = 100

    trigger_vals = [1,2,4,5]
    path = r"..\..\user_access\raw"
    filename = f'{SID}_{SDATE}_1.npy'
    full_path = os.path.join(path, filename)
    dataAcquired = np.load(full_path)

    latency,erp_std_fz,erp_dev_fz,erp_std_cz,erp_dev_cz,erp_std_pz,erp_dev_pz,erp_con_fz,erp_inc_fz,erp_con_cz,erp_inc_cz,erp_con_pz,erp_inc_pz = post_process_onebatch(dataAcquired,fs,clip,line,low,high,trigger_vals)
    
    generate_report_pdf(latency,
                        erp_std_fz,erp_dev_fz,
                        erp_std_cz,erp_dev_cz,
                        erp_std_pz,erp_dev_pz,
                        erp_con_fz,erp_inc_fz,
                        erp_con_cz,erp_inc_cz,
                        erp_con_pz,erp_inc_pz)



with open(r"..\assets\utils\biodata.txt", "r") as file:
    lines = [line.strip() for line in file.readlines()]

SID = lines[1].replace("ID:","")
SDATE = lines[4].replace("Date:","")
location = lines[12].replace("Location:","")
if location == "America":
    line = 60
else:
    line = 50
# print("line: ",line)
low,high = 0.1,10
fs = 500
clip = 100

trigger_vals = [1,2,4,5]
path = r"..\..\user_access\raw"
filename_1, filename_2, filename_3 = f'{SID}_{SDATE}_1.npy', f'{SID}_{SDATE}_2.npy', f'{SID}_{SDATE}_3.npy'
full_path_1,full_path_2,full_path_3 = os.path.join(path, filename_1), os.path.join(path, filename_2), os.path.join(path, filename_3)
dataAcquired_1 = np.load(full_path_1)
dataAcquired_2 = np.load(full_path_2)
dataAcquired_3 = np.load(full_path_3)

print("dataAcquired_1.shape: ",dataAcquired_1.shape)
latency,erp_std_fz_1,erp_dev_fz_1,erp_std_cz_1,erp_dev_cz_1,erp_std_pz_1,erp_dev_pz_1,erp_con_fz_1,erp_inc_fz_1,erp_con_cz_1,erp_inc_cz_1,erp_con_pz_1,erp_inc_pz_1 = post_process_onebatch(dataAcquired_1,fs,clip,line,low,high,trigger_vals)  
print("dataAcquired_2.shape: ",dataAcquired_2.shape)
latency,erp_std_fz_2,erp_dev_fz_2,erp_std_cz_2,erp_dev_cz_2,erp_std_pz_2,erp_dev_pz_2,erp_con_fz_2,erp_inc_fz_2,erp_con_cz_2,erp_inc_cz_2,erp_con_pz_2,erp_inc_pz_2 = post_process_onebatch(dataAcquired_2,fs,clip,line,low,high,trigger_vals)
print("dataAcquired_3.shape: ",dataAcquired_3.shape)
latency,erp_std_fz_3,erp_dev_fz_3,erp_std_cz_3,erp_dev_cz_3,erp_std_pz_3,erp_dev_pz_3,erp_con_fz_3,erp_inc_fz_3,erp_con_cz_3,erp_inc_cz_3,erp_con_pz_3,erp_inc_pz_3 = post_process_onebatch(dataAcquired_3,fs,clip,line,low,high,trigger_vals)

# Average ERPs
erp_std_fz = np.nanmean(np.array([erp_std_fz_1, erp_std_fz_2, erp_std_fz_3]), axis=0)
erp_dev_fz = np.nanmean(np.array([erp_dev_fz_1, erp_dev_fz_2, erp_dev_fz_3]), axis=0)
erp_std_cz = np.nanmean(np.array([erp_std_cz_1, erp_std_cz_2, erp_std_cz_3]), axis=0)
erp_dev_cz = np.nanmean(np.array([erp_dev_cz_1, erp_dev_cz_2, erp_dev_cz_3]), axis=0)
erp_std_pz = np.nanmean(np.array([erp_std_pz_1, erp_std_pz_2, erp_std_pz_3]), axis=0)
erp_dev_pz = np.nanmean(np.array([erp_dev_pz_1, erp_dev_pz_2, erp_dev_pz_3]), axis=0)
erp_con_fz = np.nanmean(np.array([erp_con_fz_1, erp_con_fz_2, erp_con_fz_3]), axis=0)
erp_inc_fz = np.nanmean(np.array([erp_inc_fz_1, erp_inc_fz_2, erp_inc_fz_3]), axis=0)
erp_con_cz = np.nanmean(np.array([erp_con_cz_1, erp_con_cz_2, erp_con_cz_3]), axis=0)
erp_inc_cz = np.nanmean(np.array([erp_inc_cz_1, erp_inc_cz_2, erp_inc_cz_3]), axis=0)
erp_con_pz = np.nanmean(np.array([erp_con_pz_1, erp_con_pz_2, erp_con_pz_3]), axis=0)
erp_inc_pz = np.nanmean(np.array([erp_inc_pz_1, erp_inc_pz_2, erp_inc_pz_3]), axis=0)


generate_report_pdf(latency,
                    erp_std_fz,erp_dev_fz,
                    erp_std_cz,erp_dev_cz,
                    erp_std_pz,erp_dev_pz,
                    erp_con_fz,erp_inc_fz,
                    erp_con_cz,erp_inc_cz,
                    erp_con_pz,erp_inc_pz)
